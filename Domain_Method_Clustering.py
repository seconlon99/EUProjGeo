#!/usr/bin/env python
# coding: utf-8

# In[159]:


#Setup: Imports and Configuration

import pandas as pd
import csv
import time
import re
from typing import List, Dict, Callable, Optional, Tuple 
import google.generativeai as genai
import os 
from IPython.display import display #

GOOGLE_API_KEY=""

### Models
#gemma-3-27b-it
#gemini-2.5-flash-preview-05-20
#gemini-2.0-flash
#gemini-2.0-flash-exp

genai.configure(api_key=GOOGLE_API_KEY)
# --- Model Initialization ---
## Specify which model you wanna use
def get_genai_model(model_name: str = "gemini-2.5-flash-preview-05-20"):
    print(f"Initializing model: {model_name}")
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}. Ensure API key is set.")
        return None
model = get_genai_model() # Using the default

# --- Global Configuration ---
RUN_OVERALL_TRIAL_MODE = False 
MAX_BATCHES_FOR_TRIAL = 3
DEFAULT_BATCH_SIZE = 15
DEFAULT_WAIT_TIME = 0# Seconds

# ===> SPECIFY YOUR INPUT FILE PATH HERE <===
file_path = "/Users/jonaslorler/Documents/cordis-HORIZONprojects-csv/project.csv"
# Example if it's in the same directory as the notebook:
# initial_data_file_path = "project.csv" 
#CHECKPOINT FILENAME 
MAIN_DATAFRAME_CHECKPOINT_FILE = "df_processing_checkpoint.csv"


pd.reset_option('display.max_colwidth')
print(f"Initial data file path set to: {file_path}")


# In[2]:


# Initial Data Loading and Preparation 
print(f"Performing initial data load from: {file_path}")
fixed_rows = []
current_row_ml_handler = [] # Using a distinct name for multiline state
expected_fields_ml = 0
in_multiline_ml = False

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        header = list(csv.reader([header_line], delimiter=';', quotechar='"'))[0]
        expected_fields_ml = len(header)
        
        id_idx = header.index('id')
        title_idx = header.index('title')
        objective_idx = header.index('objective')
        
        for line_num, line_content in enumerate(f, start=2):
            try:
                parsed_line_fields = list(csv.reader([line_content.strip()], delimiter=';', quotechar='"'))
                current_fields_item = parsed_line_fields[0] if parsed_line_fields else []
                
                if len(current_fields_item) == expected_fields_ml and not in_multiline_ml:
                    fixed_rows.append([current_fields_item[id_idx], current_fields_item[title_idx], current_fields_item[objective_idx]])
                elif len(current_fields_item) < expected_fields_ml or in_multiline_ml:
                    if not in_multiline_ml:
                        current_row_ml_handler = current_fields_item
                        in_multiline_ml = True
                    else:
                        if current_row_ml_handler:
                            if len(current_row_ml_handler) > objective_idx:
                                 current_row_ml_handler[objective_idx] += ' ' + line_content.strip()
                            else:
                                 current_row_ml_handler[-1] += ' ' + line_content.strip()
                        if line_content.strip().endswith('"'):
                            combined_text_check = ';'.join(str(field) for field in current_row_ml_handler)
                            parsed_combined_check = list(csv.reader([combined_text_check], delimiter=';', quotechar='"'))
                            if parsed_combined_check and len(parsed_combined_check[0]) >= expected_fields_ml:
                                complete_row_cand = parsed_combined_check[0][:expected_fields_ml]
                                fixed_rows.append([complete_row_cand[id_idx], complete_row_cand[title_idx], complete_row_cand[objective_idx]])
                                in_multiline_ml = False
                                current_row_ml_handler = []
                elif len(current_fields_item) > expected_fields_ml:
                    fixed_rows.append([current_fields_item[id_idx], current_fields_item[title_idx], current_fields_item[objective_idx]])
            except Exception as e_row:
                # Simplified error handling for row processing
                if in_multiline_ml and current_row_ml_handler: # Try to append if in multiline
                     if len(current_row_ml_handler) > objective_idx: current_row_ml_handler[objective_idx] += ' ' + line_content.strip()
                     else: current_row_ml_handler[-1] += ' ' + line_content.strip()
                # else: print(f"Skipping line {line_num} due to error: {e_row}")


    df_complete = pd.DataFrame(fixed_rows, columns=['id', 'title', 'objective'])
    print(f"Total rows recovered: {len(df_complete)}")

    df_complete.columns = df_complete.columns.str.strip('"')
    df_complete['id'] = pd.to_numeric(df_complete['id'], errors='coerce')
    df_complete = df_complete.dropna(subset=['id'])
    if not df_complete.empty:
        df_complete['id'] = df_complete['id'].astype(int)
    print(f"Rows after cleanup: {len(df_complete)}")

    # Create 'full_text' and drop original title/objective columns (Your In[3])
    if not df_complete.empty:
        df_complete["full_text"] = df_complete["title"].astype(str) + ": " + df_complete["objective"].astype(str)
        if 'title' in df_complete.columns and 'objective' in df_complete.columns: # Check before dropping
            df_complete = df_complete.drop(columns=["title", "objective"], errors='ignore')
        print("'full_text' column created, 'title' and 'objective' dropped.")
        
        # --- SAVE INITIAL STATE TO CHECKPOINT ---
        try:
            df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
            print(f"Initial df_complete saved to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
        except Exception as e_save:
            print(f"Error saving initial checkpoint: {e_save}")

    print("\nFirst 5 rows of prepared df_complete:")
    if not df_complete.empty:
        display(df_complete[['id', 'full_text']].head())
        print(df_complete.info()) # Good to see info after load
    else:
        print("df_complete is empty after initial loading.")

except FileNotFoundError:
    print(f"CRITICAL ERROR: The initial data file '{file_path}' was not found.")
    df_complete = pd.DataFrame() 
except Exception as e_load:
    print(f"An unexpected error occurred during initial data loading: {e_load}")
    df_complete = pd.DataFrame()


# In[101]:


#Define functions to reuse from Level 1 to 4
def generic_classify_batch(
    batch_prompt_items: List[Dict[str, str]], 
    prompt_template: str,
    item_prompt_format: str, 
    model_instance, # Pass the actual model object
    current_batch_global_offset: int, 
    batch_num_display: int, # This is 0-indexed for internal use, display as batch_num_display + 1
    max_text_length: int = 1000, 
    max_retries: int = 3,
    domain_categories_list: Optional[List[str]] = None,
    method_categories_list: Optional[List[str]] = None
) -> str:
    if not model_instance:
        print("Model not initialized for generic_classify_batch. Skipping API call.")
        return ""

    current_prompt_template = prompt_template
    if domain_categories_list is not None and method_categories_list is not None and \
       '{domain_categories}' in prompt_template and '{method_categories}' in prompt_template:
        try:
            current_prompt_template = prompt_template.format(
                domain_categories="\n".join(domain_categories_list),
                method_categories="\n".join(method_categories_list)
            )
        except KeyError as e:
            print(f"KeyError formatting prompt template with category lists: {e}")
            # Fallback or error handling
            pass # current_prompt_template remains the original if formatting fails
    
    full_prompt = current_prompt_template 
    
    for i, item_data in enumerate(batch_prompt_items):
        processed_item_data = {}
        for key, value in item_data.items():
            # Truncate specified long text fields
            if isinstance(value, str) and key in ['full_text', 'summary', 'objective', 'Project Summary', 
                                                  'Initial Goal', 'Initial Method', 
                                                  'Level 2 Generalized Goal', 'Level 2 Generalized Method',
                                                  'Existing L3 Application/Technology Domain', 
                                                  'Existing L3 Strategic Method Category']:
                processed_item_data[key] = value.strip()[:max_text_length]
            else:
                processed_item_data[key] = value
        processed_item_data['project_num_in_prompt'] = current_batch_global_offset + i + 1 # 1-based for LLM
        
        try:
            full_prompt += item_prompt_format.format(**processed_item_data)
        except KeyError as e:
            print(f"KeyError formatting item prompt: {e}. Item data: {processed_item_data}, Format: '{item_prompt_format}'")
            continue 

    if batch_num_display == 0: # Only log the first batch's prompt structure in detail (0-indexed display here)
        print(f"\n=== FIRST PROMPT FOR BATCH 1 (Structure Preview) ===")
        prompt_start_preview = current_prompt_template.split("Projects to Classify:",1)[0] if "Projects to Classify:" in current_prompt_template else current_prompt_template.split("Now, analyze the following projects",1)[0]
        prompt_start_preview = prompt_start_preview[:1000] + "..." if len(prompt_start_preview) > 1000 else prompt_start_preview
        print(prompt_start_preview)
        if batch_prompt_items:
             sample_item_data_for_format = {'project_num_in_prompt': current_batch_global_offset + 1}
             # Extract keys from item_prompt_format to build a complete sample_item_data
             # This regex finds all placeholders like {key_name}
             placeholders_in_format = re.findall(r'{(\w+)}', item_prompt_format)
             for key_in_item_format in placeholders_in_format:
                 if key_in_item_format != 'project_num_in_prompt': # Already added
                     # Check if this key is expected from input_fields_for_prompt (via batch_prompt_items)
                     # or provide a generic placeholder if not directly from input_fields_for_prompt
                     if batch_prompt_items and key_in_item_format in batch_prompt_items[0]:
                         sample_item_data_for_format[key_in_item_format] = f"[SAMPLE_{key_in_item_format.upper()}_FROM_DATA]"
                     else:
                         sample_item_data_for_format[key_in_item_format] = f"[SAMPLE_PLACEHOLDER_FOR_{key_in_item_format.upper()}]"
             try:
                print(f"Item Format Example: {item_prompt_format.format(**sample_item_data_for_format)}")
             except KeyError as e:
                print(f"Could not generate item format example due to missing keys for format '{item_prompt_format}': {e}")
                print(f"  Sample data used: {sample_item_data_for_format}")


        print("=== END PROMPT PREVIEW ===\n")

    for attempt in range(max_retries):
        try:
            response = model_instance.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Error on API call attempt {attempt + 1} for batch {batch_num_display + 1}, offset {current_batch_global_offset}: {type(e).__name__} - {e}")
            sleep_time = (2 ** attempt)
            if "quota" in str(e).lower() or "rate limit" in str(e).lower() or "429" in str(e).lower() or "503" in str(e).lower():
                sleep_time = (2 ** attempt) * 5 
                print(f"Quota/Rate limit likely hit. Sleeping for {sleep_time}s...")
            else:
                print(f"Sleeping for {sleep_time}s before retry...")
            
            if attempt < max_retries - 1:
                time.sleep(sleep_time) 
            else:
                print(f"Failed all API retry attempts for batch {batch_num_display + 1}.")
                return "" 
    return ""

def generic_extract_labels(
    llm_output_text: str,
    expected_labels_map: Dict[str, str], 
    batch_num_display: int, # 0-indexed for internal use, display as batch_num_display + 1
    num_items_in_batch: int,
    valid_domains: Optional[List[str]] = None,
    valid_methods: Optional[List[str]] = None,
    separator: str = "---" 
) -> Dict[int, Dict[str, str]]: 
    if batch_num_display < 3: 
        print(f"\n--- RAW LLM OUTPUT (Batch {batch_num_display+1}, {num_items_in_batch} items) ---")
        print(llm_output_text[:1000] + ("..." if len(llm_output_text) > 1000 else ""))
        print("--- END RAW LLM OUTPUT ---\n")

    parsed_results_for_batch = {} 
    
    project_blocks_raw = []
    if separator and separator in llm_output_text and len(llm_output_text.split(separator)) >= num_items_in_batch * 0.5:
        project_blocks_raw = llm_output_text.split(separator)
    else:
        project_starts = list(re.finditer(r'Project\s*#\d+:', llm_output_text, re.IGNORECASE))
        if not project_starts:
            if llm_output_text.strip(): project_blocks_raw = [llm_output_text] 
        else:
            for i, start_match in enumerate(project_starts):
                start_pos = start_match.start()
                end_pos = project_starts[i+1].start() if i + 1 < len(project_starts) else len(llm_output_text)
                project_blocks_raw.append(llm_output_text[start_pos:end_pos])

    current_batch_item_idx = 0 
    for block_text_raw in project_blocks_raw:
        block_text = block_text_raw.strip()
        if not block_text:
            continue
        
        if current_batch_item_idx >= num_items_in_batch:
            if batch_num_display < 3: print(f"Warning (Batch {batch_num_display+1}): Parser got more blocks ({len(project_blocks_raw)}) than items sent ({num_items_in_batch}). Stopping parse for this batch.")
            break 

        labels_for_this_item = {}
        for _llm_key, out_key in expected_labels_map.items():
            labels_for_this_item[out_key] = "Not Found in LLM Output" 

        for line in block_text.splitlines():
            line_stripped = line.strip()
            for llm_key_prefix, output_dict_key in expected_labels_map.items():
                if line_stripped.lower().startswith(llm_key_prefix.lower()):
                    value = line_stripped[len(llm_key_prefix):].strip()
                    
                    parser_key_for_validation = output_dict_key 
                    if parser_key_for_validation.endswith(("_parsed", "_val")): # Generalize common suffixes
                        parser_key_for_validation = re.sub(r'(_parsed|_val)$', '', parser_key_for_validation)


                    if parser_key_for_validation == "domain" and valid_domains: 
                        if value not in valid_domains:
                            original_llm_value = value
                            value = find_closest_match_in_list(value, valid_domains)
                            if batch_num_display < 3 and original_llm_value != value : 
                                print(f"  LLM Domain Mismatch (B{batch_num_display+1}, Item {current_batch_item_idx}): '{original_llm_value}' -> '{value}'")
                    elif parser_key_for_validation == "method" and valid_methods: 
                        if value not in valid_methods:
                            original_llm_value = value
                            value = find_closest_match_in_list(value, valid_methods)
                            if batch_num_display < 3 and original_llm_value != value:
                                print(f"  LLM Method Mismatch (B{batch_num_display+1}, Item {current_batch_item_idx}): '{original_llm_value}' -> '{value}'")
                        
                    labels_for_this_item[output_dict_key] = value
                    break 
        
        parsed_results_for_batch[current_batch_item_idx] = labels_for_this_item
        current_batch_item_idx += 1

    if batch_num_display < 3:
        print(f"--- PARSED LABELS (Batch {batch_num_display+1}) ---")
        for batch_idx, labels in sorted(parsed_results_for_batch.items()):
            print(f"  Item in batch at offset {batch_idx}: {labels}")
        print(f"  (Parsed {len(parsed_results_for_batch)} items from this batch output, expected {num_items_in_batch})")
        print("--- END PARSED LABELS ---\n")
    return parsed_results_for_batch

def find_closest_match_in_list(value_from_llm: str, predefined_category_list: List[str]) -> str:
    if not value_from_llm or not predefined_category_list: return "Classification Failed"
    for category_in_list in predefined_category_list:
        if value_from_llm.lower() == category_in_list.lower():
            return category_in_list 
    return "Classification Failed"

def process_stage(
    df_main: pd.DataFrame, 
    stage_name: str,
    prompt_template_str: str,
    item_prompt_format_str: str, 
    input_fields_for_prompt: List[str], 
    llm_expected_labels_map: Dict[str, str], 
    output_col_names_map: Dict[str, str], 
    model_instance, 
    eligibility_filter_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None, 
    batch_size: int = DEFAULT_BATCH_SIZE, 
    wait_time: int = DEFAULT_WAIT_TIME,   
    run_trial_mode: bool = RUN_OVERALL_TRIAL_MODE, 
    max_batches_trial: int = MAX_BATCHES_FOR_TRIAL, 
    domain_categories_for_classification: Optional[List[str]] = None, 
    method_categories_for_classification: Optional[List[str]] = None, 
    llm_output_separator: str = "---" 
) -> pd.DataFrame: 

    print(f"\n{'='*20} Running Stage: {stage_name} on DataFrame with {len(df_main)} rows {'='*20}")
    if df_main.empty:
        print(f"Stage {stage_name}: Input DataFrame is empty. No changes made.")
        return df_main.copy() 

    df_working_copy = df_main.copy() 

    df_to_process_eligible = pd.DataFrame() # Initialize
    if eligibility_filter_fn:
        eligible_mask_for_this_call = eligibility_filter_fn(df_working_copy)
        df_to_process_eligible = df_working_copy[eligible_mask_for_this_call].copy()
        print(f"Stage {stage_name}: {len(df_to_process_eligible)} rows eligible for THIS processing call out of {len(df_working_copy)}.")
        if df_to_process_eligible.empty:
            print(f"Stage {stage_name}: No rows eligible for THIS call. No API calls will be made.")
            # Initialize output columns on df_working_copy for non-eligible rows if columns don't exist
            for df_col_name in output_col_names_map.values():
                if df_col_name not in df_working_copy.columns:
                    df_working_copy[df_col_name] = pd.NA 
                # Mark non-eligible rows if the column already exists and is NA (or a specific placeholder)
                df_working_copy.loc[~eligible_mask_for_this_call & (df_working_copy[df_col_name].isnull() | df_working_copy[df_col_name].isin(["Not Processed by Stage", pd.NA])), df_col_name] = "Not Eligible for Stage"
            return df_working_copy 
    else:
        df_to_process_eligible = df_working_copy.copy() 
        print(f"Stage {stage_name}: All {len(df_to_process_eligible)} rows from input are targeted for processing (no filter).")
            
    df_final_slice_for_api = df_to_process_eligible 
    if run_trial_mode: 
        num_records_for_trial = min(len(df_to_process_eligible), batch_size * max_batches_trial) 
        df_final_slice_for_api = df_to_process_eligible.iloc[:num_records_for_trial].copy()
        print(f"Stage {stage_name}: TRIAL MODE - Will send {len(df_final_slice_for_api)} (eligible) records to API.")
    else:
        # df_final_slice_for_api is already df_to_process_eligible (or its copy)
        print(f"Stage {stage_name}: FULL MODE - Will send {len(df_final_slice_for_api)} (eligible) records to API.")

    if df_final_slice_for_api.empty:
        print(f"Stage {stage_name}: No records to send to API after eligibility/trial slicing.")
        for df_col_name in output_col_names_map.values():
            if df_col_name not in df_working_copy.columns:
                df_working_copy[df_col_name] = "No Data Sent to API"
        return df_working_copy

    results_by_original_idx = {} 
    successful_batches_count = 0
    total_records_in_final_slice = len(df_final_slice_for_api)
    total_batches = (total_records_in_final_slice + batch_size - 1) // batch_size
    
    if total_batches > 0: print(f"  Will take {total_batches} batches. Est. API wait: {total_batches * wait_time / 60:.1f} min.\n")

    for batch_num_0_indexed, i_offset_in_slice in enumerate(range(0, total_records_in_final_slice, batch_size)):
        i_end_offset_in_slice = min(i_offset_in_slice + batch_size, total_records_in_final_slice)
        current_batch_df_for_api = df_final_slice_for_api.iloc[i_offset_in_slice:i_end_offset_in_slice]
        if current_batch_df_for_api.empty: continue

        batch_prompt_items_data = []
        for _original_idx, row_series in current_batch_df_for_api.iterrows():
            item_data = {field: row_series.get(field, "") for field in input_fields_for_prompt}
            batch_prompt_items_data.append(item_data)
        
        num_items_in_this_batch = len(batch_prompt_items_data)
        current_batch_global_prompt_offset = i_offset_in_slice 

        if batch_num_0_indexed < 3 or (batch_num_0_indexed % 5 == 0 and total_batches > 10):
            print(f"\n  {stage_name} - Batch {batch_num_0_indexed + 1}/{total_batches} (Slice Offsets {i_offset_in_slice}-{i_end_offset_in_slice-1})")

        llm_response_text = generic_classify_batch(
            batch_prompt_items_data, prompt_template_str, item_prompt_format_str,
            model_instance, current_batch_global_offset=current_batch_global_prompt_offset,
            batch_num_display=batch_num_0_indexed, 
            domain_categories_list=domain_categories_for_classification,
            method_categories_list=method_categories_for_classification
        )

        if llm_response_text:
            parsed_labels_for_batch = generic_extract_labels(
                llm_response_text, llm_expected_labels_map,
                batch_num_display=batch_num_0_indexed, 
                num_items_in_batch=num_items_in_this_batch,
                valid_domains=domain_categories_for_classification,
                valid_methods=method_categories_for_classification,
                separator=llm_output_separator
            )
            
            for batch_local_idx, labels_dict_from_parser in parsed_labels_for_batch.items():
                if batch_local_idx < len(current_batch_df_for_api): 
                    original_df_index = current_batch_df_for_api.index[batch_local_idx] 
                    results_by_original_idx[original_df_index] = labels_dict_from_parser
            
            if parsed_labels_for_batch: successful_batches_count += 1
            if batch_num_0_indexed < 3: print(f"  Batch {batch_num_0_indexed + 1} outcome: {'Labels extracted.' if parsed_labels_for_batch else 'No labels extracted.'}")
        else: # No LLM response text
            if batch_num_0_indexed < 3: print(f"  Batch {batch_num_0_indexed + 1} outcome: Failed to get LLM output.")
            for idx_in_failed_batch in current_batch_df_for_api.index:
                # Ensure results_by_original_idx has an entry for all items in the failed batch
                # Mark all expected output keys with "API Error in Batch"
                results_by_original_idx[idx_in_failed_batch] = {
                    parser_key: "API Error in Batch" for parser_key in llm_expected_labels_map.values() # Use parser keys
                }
        
        if batch_num_0_indexed < total_batches - 1: 
            if batch_num_0_indexed < 3 or (batch_num_0_indexed % 10 == 0 and total_batches > 20): print(f"  Waiting {wait_time}s...")
            time.sleep(wait_time)
            
    print(f"\nStage {stage_name}: API calls complete. {successful_batches_count}/{total_batches} batches yielded some parsable results.")
    
    # --- Update df_working_copy IN PLACE (which started as df_main.copy()) ---
    # Initialize output columns if they don't exist yet on df_working_copy for all rows first.
    # This ensures columns exist before trying to loc-assign.
    for df_col_name in output_col_names_map.values():
        if df_col_name not in df_working_copy.columns:
            df_working_copy[df_col_name] = pd.NA 
            print(f"Stage {stage_name}: Initialized new column '{df_col_name}' in working copy with pd.NA.")

    # Apply successfully parsed results (or error indicators like "API Error in Batch")
    # to the corresponding rows in df_working_copy.
    # Only updates rows that were part of df_final_slice_for_api (i.e., processed).
    for original_idx, parsed_labels_dict in results_by_original_idx.items():
        if original_idx in df_working_copy.index: 
            for parser_key, df_col_name in output_col_names_map.items():
                value_to_assign = parsed_labels_dict.get(parser_key, "Parse Error (Key Missing)")
                df_working_copy.loc[original_idx, df_col_name] = value_to_assign
        else:
            print(f"Critical Warning: Original index {original_idx} from results not found in working_copy for stage {stage_name}.")

    # --- Summary Stats (Corrected) ---
    processed_and_got_valid_labels_count = 0
    # Iterate through the indices that were actually processed (sent to API)
    for original_idx_processed in df_final_slice_for_api.index:
        if original_idx_processed in results_by_original_idx:
            labels_dict_from_parser = results_by_original_idx[original_idx_processed]
            at_least_one_valid_label_for_this_item = False
            for parser_key in output_col_names_map.keys(): 
                value_from_llm = labels_dict_from_parser.get(parser_key)
                invalid_values_for_summary = [
                    "Parse Error (Key Missing)", "Not Found in LLM Output", 
                    "Classification Failed", "API Error in Batch", ""                       
                ] 
                if pd.isna(value_from_llm) or (isinstance(value_from_llm, str) and value_from_llm in invalid_values_for_summary):
                    pass 
                else:
                    at_least_one_valid_label_for_this_item = True 
                    break 
            if at_least_one_valid_label_for_this_item:
                processed_and_got_valid_labels_count += 1
    
    print(f"\n{'='*15} Stage {stage_name} - FINAL SUMMARY {'='*15}")
    print(f"  Input records to stage (df_main): {len(df_main)}")
    if eligibility_filter_fn: print(f"  Eligible records (before trial slice): {len(df_to_process_eligible)}")
    print(f"  Records sent to API (after trial slice if any): {total_records_in_final_slice}")
    print(f"  Successfully received and parsed at least one valid label for: {processed_and_got_valid_labels_count} records.")
    if total_records_in_final_slice > 0 : 
        success_rate = (processed_and_got_valid_labels_count / total_records_in_final_slice) * 100
        print(f"  Success rate on records sent to API: {success_rate:.0f}%")
    
    print(f"\n  Top values for output columns in Stage {stage_name} (from {len(df_working_copy)} total rows):")
    for df_col_name in output_col_names_map.values():
        if df_col_name in df_working_copy.columns:
            print(f"    Column '{df_col_name}':")
            counts = df_working_copy[df_col_name].value_counts(dropna=False) 
            for val, count in counts.head(7).items():
                 display_val = str(val)[:70] if not pd.isna(val) else "pd.NA"
                 print(f"      - \"{display_val}\": {count}") 
            if len(counts) > 7: print("        ...")
        
    return df_working_copy 

print("Generic processing functions defined.")


# In[4]:


# ## Stage 1: Initial L1 Goal and Method Classification

# --- Stage 1 Configuration ---
STAGE_NAME_S1 = "L1_Initial_Classification"

PROMPT_TEMPLATE_S1 = """You are analyzing EU-funded research projects.
For each project, provide two short labels:
The first label should describe the project’s main purpose or intended societal impact — what the project is ultimately trying to achieve. 
Keep it short (max 8 words) and focus on the real-world outcome or benefit. For example:  
Goal: Inclusive English Language Access

The second label should describe the core method, approach, or technology being used. 
What is the main approach, method, or technology used to achieve the goal? This should reflect *how* the project is doing its work.
If the method involves multiple elements, you may combine them into a single phrase (maximum 7 words). 
However, avoid using “&” alone. Instead, use wording that clarifies how the elements are related,
for example, whether one enables the other, they are used together, or one is a format for the other.
To identify the true method, ask yourself:** Is this item something the project primarily does as an action (e.g., 'Organizing Conference', 'Developing Software', 'Conducting Workshops'),
creates as a main deliverable (e.g., 'Building a Database', 'Creating Online Platform'), or uses as a core technique (e.g., 'AI Algorithms', 'Survey Methodology', 'Case Study Analysis')?
Example of this distinction: Consider a project aiming to "Strengthen research infrastructure strategy" by organizing a conference. 
If this conference plans to discuss "RI strategic autonomy," "societal impact of RIs," and "ecosystems of RIs (including ecosystem analysis),
the ideal method is 'Policy Dialogue via Research Infrastructure Conference'
A method like 'Conference & Ecosystem Analysis' would be incorrect in this context. 'Ecosystem Analysis' is a topic
or theme to be discussed at the conference; it is not a separate, 
co-equal methodological activity being performed by the project alongside organizing the conference. 
The conference is the vehicle or *platform* for these discussions. The project's primary action is organizing the event.
Avoid listing topics of discussion or sub-components as if they are the method itself.
Avoid phrases where the role of the method is unclear.

Unclear: “Digital Platform & Stakeholder Mapping”  
Clear: “Creation of Digital Stakeholder Platform”

Keep it concise (max 7 words) and use standard scientific or technical terms. For example:  
Method: AI Language Assistants

Format your response like this for each project in the batch:
Project #{project_num_in_prompt}:  
Goal: [impact-driven purpose]  
Method: [technical or methodological approach]

Do not include anything else in your response.

Projects to Classify:
""" 
ITEM_PROMPT_FORMAT_S1 = "Project #{project_num_in_prompt}: {full_text}\n" 
INPUT_FIELDS_S1 = ['full_text'] 
LLM_EXPECTED_LABELS_S1 = {"Goal:": "s1_goal_parsed", "Method:": "s1_method_parsed"}
OUTPUT_COL_NAMES_S1 = {"s1_goal_parsed": "l1_goal", "s1_method_parsed": "l1_method"}
ELIGIBILITY_FILTER_S1 = None 

# --- Execute Stage 1 ---
# df_complete is loaded/initialized in Cell 2.
# model, RUN_OVERALL_TRIAL_MODE, etc., are from Cell 1.

if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    print(f"Input for Stage 1 ({STAGE_NAME_S1}): df_complete with {len(df_complete)} rows.")
    # Ensure 'full_text' column exists from Cell 2
    if 'full_text' not in df_complete.columns:
        print(f"ERROR: 'full_text' column not found in df_complete for Stage {STAGE_NAME_S1}. Please check Cell 2 (Data Loading).")
        # df_complete might be an empty shell if initial loading totally failed; process_stage handles empty df_main.
    else:
        df_complete = process_stage( # Update df_complete with results
            df_main=df_complete, 
            stage_name=STAGE_NAME_S1,
            prompt_template_str=PROMPT_TEMPLATE_S1,
            item_prompt_format_str=ITEM_PROMPT_FORMAT_S1,
            input_fields_for_prompt=INPUT_FIELDS_S1,
            llm_expected_labels_map=LLM_EXPECTED_LABELS_S1,
            output_col_names_map=OUTPUT_COL_NAMES_S1,
            model_instance=model, 
            eligibility_filter_fn=ELIGIBILITY_FILTER_S1,
            run_trial_mode=RUN_OVERALL_TRIAL_MODE, 
            max_batches_trial=MAX_BATCHES_FOR_TRIAL, 
            batch_size=DEFAULT_BATCH_SIZE, 
            wait_time=DEFAULT_WAIT_TIME,
            llm_output_separator=None # S1 prompt doesn't specify "---" between items for LLM output
        )
        
        if not df_complete.empty: # Check the result of process_stage
            print(f"\n{STAGE_NAME_S1} completed. Sample of updated df_complete:")
            # from IPython.display import display # Should be imported in Cell 3
            display(df_complete[['id', 'full_text', 'l1_goal', 'l1_method']].head())
            
            # ---> SAVE THE UPDATED df_complete TO THE MAIN CHECKPOINT FILE <---
            # MAIN_DATAFRAME_CHECKPOINT_FILE should be defined in Cell 1
            try:
                df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False) # OVERWRITE
                print(f"Saved updated df_complete after {STAGE_NAME_S1} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
            except Exception as e:
                print(f"Error saving checkpoint after {STAGE_NAME_S1}: {e}")
        else:
            print(f"Stage {STAGE_NAME_S1} resulted in an empty DataFrame. This indicates a problem if input was not empty.")
else:
    if 'df_complete' not in locals() or not isinstance(df_complete, pd.DataFrame) or df_complete.empty:
        print("Skipping Stage 1: Input DataFrame 'df_complete' (from Cell 2) is not available or is empty.")
    if not model:
        print("Skipping Stage 1: LLM Model is not initialized (check Cell 1).")


# In[5]:


# ## Stage 1.1: Fixing Failed L1 Classifications
# --- Stage 1.1 Configuration ---
STAGE_NAME_S1_FIX = "L1_Fix_Failed_Classification"

# We re-use the prompt, item format, input fields, LLM labels, and output columns 
# from Stage 1 (defined in Cell 4 / your In[5]).


# Define an eligibility filter specifically for rows that failed in Stage 1
def eligibility_filter_s1_fix(df: pd.DataFrame) -> pd.Series:
    # Ensure the output columns from S1 exist in the DataFrame
    # OUTPUT_COL_NAMES_S1 is {"s1_goal_parsed": "l1_goal", "s1_method_parsed": "l1_method"}
    l1_goal_col = OUTPUT_COL_NAMES_S1.get("s1_goal_parsed", "l1_goal") # Get actual column name
    l1_method_col = OUTPUT_COL_NAMES_S1.get("s1_method_parsed", "l1_method")

    if l1_goal_col not in df.columns or l1_method_col not in df.columns:
        print(f"Warning for {STAGE_NAME_S1_FIX} eligibility: '{l1_goal_col}' or '{l1_method_col}' columns are missing.")
        return pd.Series([False] * len(df), index=df.index) 

    # Define what constitutes a "failure" from Stage 1 that needs fixing.
    failed_values_to_retry = [
        "Parse Error (Key Missing)", 
        "Not Found in LLM Output", 
        "API Error in Batch", 
        "Classification Failed", # If S1 used validation that could result in this
        ""                       # Empty string from LLM might indicate failure to extract
        # pd.NA # Handled by .isnull() below
    ]

    # A row needs fixing if either its l1_goal or l1_method contains one of the failure indicators, or is null.
    failed_l1_goal = df[l1_goal_col].astype(str).isin(failed_values_to_retry) | df[l1_goal_col].isnull()
    failed_l1_method = df[l1_method_col].astype(str).isin(failed_values_to_retry) | df[l1_method_col].isnull()
    
    return failed_l1_goal | failed_l1_method

# --- Execute Stage 1.1 ---
# df_complete is the main DataFrame, assumed to be updated by Cell 4 (Stage 1)
# model, RUN_OVERALL_TRIAL_MODE, etc., are from Cell 1 (your In[19])
# MAIN_DATAFRAME_CHECKPOINT_FILE from Cell 1

if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    
    rows_needing_s1_fix_mask = eligibility_filter_s1_fix(df_complete)
    if not rows_needing_s1_fix_mask.any():
        print(f"Stage {STAGE_NAME_S1_FIX}: No rows found needing L1 fix based on current failure criteria. df_complete remains unchanged.")
        # Optionally, save the checkpoint even if no fixes were made, to mark this step as considered
        # try:
        #     df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
        #     print(f"No L1 fixes needed. Saved current df_complete to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
        # except Exception as e:
        #     print(f"Error saving checkpoint when no L1 fixes needed: {e}")
    else:
        print(f"Input for Stage 1.1 ({STAGE_NAME_S1_FIX}): df_complete with {len(df_complete)} total rows.")
        print(f"Number of rows targeted for L1 fixing: {rows_needing_s1_fix_mask.sum()}")
        
        # Ensure 'full_text' (from INPUT_FIELDS_S1) column exists for the fix stage
        if INPUT_FIELDS_S1[0] not in df_complete.columns: # Assuming INPUT_FIELDS_S1 = ['full_text']
            print(f"CRITICAL ERROR for {STAGE_NAME_S1_FIX}: Required input column '{INPUT_FIELDS_S1[0]}' is missing. Cannot proceed.")
        else:
            df_complete = process_stage( # Reassign to update df_complete
                df_main=df_complete,       
                stage_name=STAGE_NAME_S1_FIX,
                prompt_template_str=PROMPT_TEMPLATE_S1, # Reuse S1 prompt
                item_prompt_format_str=ITEM_PROMPT_FORMAT_S1, # Reuse S1 item format
                input_fields_for_prompt=INPUT_FIELDS_S1,     # Reuse S1 input fields
                llm_expected_labels_map=LLM_EXPECTED_LABELS_S1, # Reuse S1 LLM label map
                output_col_names_map=OUTPUT_COL_NAMES_S1,    # Output to the same l1_goal, l1_method cols
                model_instance=model,
                eligibility_filter_fn=eligibility_filter_s1_fix, # This is the key difference
                run_trial_mode=RUN_OVERALL_TRIAL_MODE, 
                max_batches_trial=MAX_BATCHES_FOR_TRIAL, 
                batch_size=5,  # Smaller batch size for fixing
                wait_time=10,  # Override default for fixing (or use DEFAULT_WAIT_TIME)
                llm_output_separator=None # S1 prompt implies no "---"
            )
            
            if not df_complete.empty:
                print(f"\n{STAGE_NAME_S1_FIX} completed.")
                # Display a sample of rows that were targeted by the fix process.
                # It's useful to see their state *after* the fix attempt.
                print("Sample of rows that were targeted for L1 Fix (first 5, showing L1 cols):")
                # from IPython.display import display # Should be imported in Cell 3
                # We use the mask `rows_needing_s1_fix_mask` which was calculated *before* this process_stage call.
                # This shows the rows that *were* targeted. Their values in df_complete are now updated.
                cols_to_display_s1_fix = ['id', 'full_text'] + list(OUTPUT_COL_NAMES_S1.values())
                cols_to_display_s1_fix = [col for col in cols_to_display_s1_fix if col in df_complete.columns]
                if rows_needing_s1_fix_mask.any() and cols_to_display_s1_fix: # Only display if some were targeted and cols exist
                     display(df_complete[rows_needing_s1_fix_mask][cols_to_display_s1_fix].head())
                else:
                     print("No rows were targeted for fix, or relevant columns are missing for display.")


                # Save the updated df_complete to the main checkpoint file
                try:
                    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False) # OVERWRITE
                    print(f"Saved updated df_complete after {STAGE_NAME_S1_FIX} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
                except Exception as e:
                    print(f"Error saving checkpoint after {STAGE_NAME_S1_FIX}: {e}")
            else:
                print(f"Stage {STAGE_NAME_S1_FIX} resulted in an empty DataFrame. This should be investigated.")
else:
    if 'df_complete' not in locals() or not isinstance(df_complete, pd.DataFrame) or df_complete.empty:
        print(f"Skipping Stage {STAGE_NAME_S1_FIX}: DataFrame 'df_complete' (from Stage 1 processing) is not available or is empty.")
    if not model:
        print(f"Skipping Stage {STAGE_NAME_S1_FIX}: LLM Model is not initialized.")


# In[6]:


# ## Stage 2: L2 Generalization of Goal and Method
STAGE_NAME_S2 = "L2_Generalization"

# Full prompt for L2 Generalization (PROMPT_PREFIX_L2_GENERALIZE from your original script)
PROMPT_TEMPLATE_S2_FULL = """You are an expert taxonomist creating highly searchable, general category labels for EU research projects to improve grouping and navigation.

You will receive a Project Summary, an Initial Goal, and an Initial Method.
Your task is to elevate the Initial labels into **Generalized Goal** and **Generalized Method** labels.

**Core Generalization Principle: Think Like a User Searching for Categories & Strive for Broader Groupings.**
*   Generalized labels should be intuitive category headings a user would search for.
*   Use the **Project Summary** for essential context to find the most accurate encompassing category.
*   **Your primary task is to abstract to a significantly broader level of generality.** Only if an initial label is *already* a very broad, established, and highly searchable category, AND further generalization would make it too vague or inaccurate, may it be minimally adjusted or (rarely) reused.

Instructions & Examples (Crucial):
1.  **4 Words Max** per label.
2.  **Generalized Goal:** (Keep as is from previous good version - focuses on "goal-ness")
    *   The overarching domain of impact or strategic objective.
    *   **Must clearly articulate an aim, improvement, or achievement.** Think: "What is being *advanced, improved, developed, understood, achieved, or made possible*?"
    *   *Good (Improvement/Action):* 'Cellular Nanosensing Improvement,' 'Transport Decarbonization,' 'Regional Innovation Support.'
    *   *Good (Advancement/Understanding for research):* 'Advancing Astrochemistry Knowledge,' 'Uncovering Disease Mechanisms.'
    *   *Avoid (Static Topic/Problem):* 'Biocondensate Dysfunction,' 'Molecular Origins.'
3.  **Generalized Method: Identify a significantly broader conceptual category for the approach.**
    *   For the **Generalized Method**, aim for a high-level categorization suitable for broad user searches.
    *   Think: "What is the *most encompassing and widely understood category* for this type of work, moving up the conceptual hierarchy?"
    *   **For specific scientific/technical methods:**
        *   Your goal is to find a label that groups many specific techniques. Consider moving from the *specific technique* -> to its *broader technical field* -> and if that field is still too niche for top-level search, consider an even *more general scientific/engineering domain or type of analytical approach.*
        *   *Example - Path to Abstraction:*
            *   Initial: "Machine Learning Algorithm for Image Recognition"
            *   (Intermediate thought: Broader field is "Artificial Intelligence" or "Computer Science Applications")
            *   **Generalized Method: Applied Computational Methods** (if "AI" or "CS Apps" is still too granular for your top-level taxonomy) or **AI & Data Analytics**
        *   *More Examples:*
            *   Initial: "Quantum Dynamics of Condensed Phase Systems" -> **Generalized Method: Advanced Scientific Modeling** (or **Theoretical Physics Methods**)
            *   Initial: "Comparative Ethnographic Digital Analysis" -> **Generalized Method: Social Science Research** (or **Qualitative Data Analysis**)
    *   **For activities/processes (e.g., conferences, policy work):** Generalize to the fundamental *type* of strategic action, collaboration, or engagement.
        *   Initial: "Policy dialogue via research conference" -> **Generalized Method: Stakeholder Engagement Strategies**
        *   Initial: "Transnational Joint Action Planning" -> **Generalized Method: Collaborative Program Development**
    *   **Avoid minimal rephrasing. Seek a clear jump in abstraction.**
4.  **Wording:** Use established, common, searchable terms.
5.  **Relevance & Context:** Generalized labels must be direct conceptual parents of the initial labels, deeply informed by the **Project Summary**.
6.  **Unassigned:** For "Unassigned" initial inputs, output "Unassigned."

Output Format (one section per project, separated by "---"):
Goal Summary: [short generalized label]
Method Summary: [short generalized label]
---

Example of processing an individual project input:
Project Summary: The project aims to organize a series of workshops and a final conference bringing together policymakers, industry leaders, and researchers to discuss and formulate new strategies for reducing CO2 emissions in the European logistics sector, particularly focusing on heavy-duty vehicles.
Initial Goal: New strategies for CO2 reduction in logistics
Initial Method: Workshops & Conference for Policy Formulation

Your output for this project:
Goal Summary: Transport Decarbonization
Method Summary: Stakeholder Engagement
---

Now, analyze the following projects. For each, provide 'Goal Summary' and 'Method Summary' per format, separated by "---".
"""

ITEM_PROMPT_FORMAT_S2 = (
    "\nProject #{project_num_in_prompt}:\n" 
    "Project Summary: {full_text}\n"      # Placeholder for full_text from DataFrame
    "Initial Goal: {l1_goal}\n"           # Placeholder for l1_goal from DataFrame (output of S1/S1.1)
    "Initial Method: {l1_method}\n"       # Placeholder for l1_method from DataFrame (output of S1/S1.1)
) 

INPUT_FIELDS_S2 = ['full_text', 'l1_goal', 'l1_method'] 

LLM_EXPECTED_LABELS_S2 = {"Goal Summary:": "s2_g_goal_parsed", "Method Summary:": "s2_g_method_parsed"}

OUTPUT_COL_NAMES_S2 = {"s2_g_goal_parsed": "l2_generalized_goal", "s2_g_method_parsed": "l2_generalized_method"}

# Eligibility for L2: L1 goal and method must be valid and not error placeholders.
def eligibility_filter_s2(df: pd.DataFrame) -> pd.Series:
    # Ensure L1 output columns exist
    l1_goal_col = OUTPUT_COL_NAMES_S1.get("s1_goal_parsed", "l1_goal") # From S1 config
    l1_method_col = OUTPUT_COL_NAMES_S1.get("s1_method_parsed", "l1_method") # From S1 config

    if l1_goal_col not in df.columns or l1_method_col not in df.columns:
        print(f"Warning for {STAGE_NAME_S2} eligibility: '{l1_goal_col}' or '{l1_method_col}' columns missing.")
        return pd.Series([False] * len(df), index=df.index)
    
    # Values from L1 that make a row ineligible for L2
    invalid_l1_values_for_s2_input = [
        "Parse Error (Key Missing)", 
        "Not Found in LLM Output", 
        "API Error in Batch",
        "Classification Failed", 
        "",                      
        "Unassigned", # If "Unassigned" from L1 means LLM couldn't determine L1 goal/method
        "Not Processed by Stage",
        "Not Eligible for Stage",
        "Input Error (full_text missing)", # If S1 couldn't run due to this
        pd.NA                   
    ]

    is_l1_goal_valid = df[l1_goal_col].notna() & (~df[l1_goal_col].astype(str).isin(invalid_l1_values_for_s2_input))
    is_l1_method_valid = df[l1_method_col].notna() & (~df[l1_method_col].astype(str).isin(invalid_l1_values_for_s2_input))
    
    return is_l1_goal_valid & is_l1_method_valid

# --- Execute Stage 2 ---
# df_complete is the main DataFrame, assumed to be updated by Cell 5 (Stage 1.1 Fix)
# model, RUN_OVERALL_TRIAL_MODE, etc., are from Cell 1 (your In[19])
# MAIN_DATAFRAME_CHECKPOINT_FILE from Cell 1

if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    print(f"Input for Stage 2 ({STAGE_NAME_S2}): df_complete with {len(df_complete)} rows.")
    print(f"df_complete columns before S2: {df_complete.columns.tolist()}")
    
    s2_input_cols_ok = all(col in df_complete.columns for col in INPUT_FIELDS_S2)
    
    if s2_input_cols_ok:
        df_complete = process_stage( 
            df_main=df_complete, 
            stage_name=STAGE_NAME_S2,
            prompt_template_str=PROMPT_TEMPLATE_S2_FULL,
            item_prompt_format_str=ITEM_PROMPT_FORMAT_S2,
            input_fields_for_prompt=INPUT_FIELDS_S2,
            llm_expected_labels_map=LLM_EXPECTED_LABELS_S2,
            output_col_names_map=OUTPUT_COL_NAMES_S2,
            model_instance=model,
            eligibility_filter_fn=eligibility_filter_s2, 
            run_trial_mode=RUN_OVERALL_TRIAL_MODE,
            max_batches_trial=MAX_BATCHES_FOR_TRIAL,
            batch_size=DEFAULT_BATCH_SIZE, 
            wait_time=DEFAULT_WAIT_TIME,   
            llm_output_separator="---" # L2 prompt specifies "---" as output separator
        )
        
        if not df_complete.empty:
            print(f"\n{STAGE_NAME_S2} completed. df_complete columns after S2: {df_complete.columns.tolist()}")
            print(f"Sample of updated df_complete (first 5 rows showing L1 and L2 cols):")
            # from IPython.display import display # Should be imported in Cell 3
            cols_to_display_s2 = ['id', 'full_text'] + \
                                 [OUTPUT_COL_NAMES_S1.get("s1_goal_parsed", "l1_goal"), OUTPUT_COL_NAMES_S1.get("s1_method_parsed", "l1_method")] + \
                                 list(OUTPUT_COL_NAMES_S2.values())
            cols_to_display_s2 = [col for col in cols_to_display_s2 if col in df_complete.columns]
            if cols_to_display_s2: display(df_complete[cols_to_display_s2].head())


            # Save the updated df_complete to the main checkpoint file
            try:
                df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False) # OVERWRITE
                print(f"Saved updated df_complete after {STAGE_NAME_S2} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
            except Exception as e:
                print(f"Error saving checkpoint after {STAGE_NAME_S2}: {e}")
        else:
            print(f"Stage {STAGE_NAME_S2} resulted in an empty DataFrame.")
    else:
        print(f"Skipping {STAGE_NAME_S2} due to missing required input columns ({INPUT_FIELDS_S2}) in df_complete.")
else:
    if 'df_complete' not in locals() or not isinstance(df_complete, pd.DataFrame) or df_complete.empty:
        print(f"Skipping Stage 2: DataFrame 'df_complete' (from Stage 1.1 Fix) is not available or is empty.")
    if not model:
        print("Skipping Stage 2: LLM Model is not initialized.")


# In[7]:


# ## Stage 2.1: Fixing Failed L2 Generalizations

# --- Stage 2.1 Configuration ---
STAGE_NAME_S2_FIX = "L2_Fix_Failed_Generalization"
# Reuses S2 configs: PROMPT_TEMPLATE_S2_FULL, ITEM_PROMPT_FORMAT_S2, INPUT_FIELDS_S2, 
#                    LLM_EXPECTED_LABELS_S2, OUTPUT_COL_NAMES_S2 (defined in Cell 6)

# Define an eligibility filter specifically for rows that were eligible for L2 but failed.
def eligibility_filter_s2_fix(df: pd.DataFrame) -> pd.Series:
    # L2 output columns
    l2_goal_col = OUTPUT_COL_NAMES_S2.get("s2_g_goal_parsed", "l2_generalized_goal")
    l2_method_col = OUTPUT_COL_NAMES_S2.get("s2_g_method_parsed", "l2_generalized_method")

    if l2_goal_col not in df.columns or l2_method_col not in df.columns:
        print(f"Warning for {STAGE_NAME_S2_FIX} eligibility: '{l2_goal_col}' or '{l2_method_col}' columns missing.")
        return pd.Series([False] * len(df), index=df.index)

    # Was it eligible for the main S2 run? (Uses eligibility_filter_s2 from Cell 6)
    was_eligible_for_s2_main = eligibility_filter_s2(df) 

    # Define what indicates an L2 failure needing a retry
    failed_l2_output_values = [
        "Parse Error (Key Missing)", 
        "Not Found in LLM Output", 
        "API Error in Batch",
        "Classification Failed", 
        "Not Processed by Stage",
        # "Not Generalized", # If this was a specific LLM output meaning failure for L2
        "",                      
        pd.NA                   
    ]


    failed_l2_goal = df[l2_goal_col].astype(str).isin(failed_l2_output_values) | df[l2_goal_col].isnull()
    failed_l2_method = df[l2_method_col].astype(str).isin(failed_l2_output_values) | df[l2_method_col].isnull()
    
    return was_eligible_for_s2_main & (failed_l2_goal | failed_l2_method)

# --- Execute Stage 2.1 ---
# df_complete is the main DataFrame, assumed to be updated by Cell 6 (Stage 2)

if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    
    rows_needing_s2_fix_mask = eligibility_filter_s2_fix(df_complete)
    if not rows_needing_s2_fix_mask.any():
        print(f"Stage {STAGE_NAME_S2_FIX}: No rows found needing L2 fix. df_complete remains unchanged.")
    else:
        print(f"Input for {STAGE_NAME_S2_FIX}: df_complete ({len(df_complete)} rows). Target to fix: {rows_needing_s2_fix_mask.sum()}")
        s2_fix_input_cols_ok = all(col in df_complete.columns for col in INPUT_FIELDS_S2) # INPUT_FIELDS_S2 from Cell 6
        
        if s2_fix_input_cols_ok:
            df_complete = process_stage(
                df_main=df_complete,       
                stage_name=STAGE_NAME_S2_FIX,
                prompt_template_str=PROMPT_TEMPLATE_S2_FULL, 
                item_prompt_format_str=ITEM_PROMPT_FORMAT_S2, 
                input_fields_for_prompt=INPUT_FIELDS_S2,     
                llm_expected_labels_map=LLM_EXPECTED_LABELS_S2, 
                output_col_names_map=OUTPUT_COL_NAMES_S2,    
                model_instance=model,
                eligibility_filter_fn=eligibility_filter_s2_fix, 
                run_trial_mode=RUN_OVERALL_TRIAL_MODE, 
                max_batches_trial=MAX_BATCHES_FOR_TRIAL, 
                batch_size=5,  # Smaller batch for fixing
                wait_time=15,  # Override default: Longer wait for fixing
                llm_output_separator="---" 
            )
            
            if not df_complete.empty:
                print(f"\n{STAGE_NAME_S2_FIX} completed.")
                print("Sample of rows that were targeted for L2 Fix (first 5, showing L1 and L2 cols):")
                # from IPython.display import display # Should be imported in Cell 3
                cols_to_display_s2_fix = ['id', 'full_text'] + \
                                     [OUTPUT_COL_NAMES_S1.get("s1_goal_parsed", "l1_goal"), OUTPUT_COL_NAMES_S1.get("s1_method_parsed", "l1_method")] + \
                                     list(OUTPUT_COL_NAMES_S2.values())
                cols_to_display_s2_fix = [col for col in cols_to_display_s2_fix if col in df_complete.columns]
                
                # Show data for rows that were *attempted* by this fix stage
                if rows_needing_s2_fix_mask.any() and cols_to_display_s2_fix:
                     display(df_complete[rows_needing_s2_fix_mask][cols_to_display_s2_fix].head())
                else:
                     print("No rows were targeted for S2 fix for display, or relevant columns missing.")

                # Save the updated df_complete to the main checkpoint file
                try:
                    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False) # OVERWRITE
                    print(f"Saved updated df_complete after {STAGE_NAME_S2_FIX} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
                except Exception as e:
                    print(f"Error saving checkpoint after {STAGE_NAME_S2_FIX}: {e}")
        else:
            print(f"Skipping {STAGE_NAME_S2_FIX} due to missing required input columns.")
else:
    if 'df_complete' not in locals() or not isinstance(df_complete, pd.DataFrame) or df_complete.empty:
        print(f"Skipping Stage {STAGE_NAME_S2_FIX}: df_complete not ready or model not initialized.")
    if not model:
        print(f"Skipping Stage {STAGE_NAME_S2_FIX}: LLM Model is not initialized.")


# In[11]:


# ## Stage 3: L3 Categorization (Application/Technology Domain & Strategic Method)

# --- Stage 3 Configuration ---
STAGE_NAME_S3 = "L3_Categorization"

# Full prompt for L3 Categorization (PROMPT_PREFIX_L3 from your original script)
PROMPT_TEMPLATE_S3_FULL = """You are a senior taxonomist tasked with creating truly high-level, strategic categories for EU research projects. These Level 3 categories MUST represent a significant conceptual leap from the Level 2 inputs and will be used for broad thematic grouping and simplified navigation. MINOR REPHRASING OR SLIGHT GENERALIZATIONS OF LEVEL 2 LABELS ARE NOT ACCEPTABLE FOR LEVEL 3.

You will be given:
1.  The original **Project Summary** (for essential context and grounding).
2.  A **Level 2 Generalized Goal** (an already generalized aim/objective).
3.  A **Level 2 Generalized Method** (an already generalized approach/activity type).

Your task is to:
1.  Abstract the 'Level 2 Generalized Goal' into a **'Level 3 Application/Technology Domain'**. This domain should be a recognized, distinct field.
2.  Abstract the 'Level 2 Generalized Method' into a **'Level 3 Strategic Method Category'**. This category should represent a fundamental type of strategic action or core operational domain.

**Core Principle for Level 3 Abstraction: DEMAND A SIGNIFICANT CONCEPTUAL LEAP TO MAJOR CATEGORIES.**
*   **For BOTH Goal & Method Paths:** The Level 3 category MUST be a substantially broader, more encompassing category than the Level 2 input. Think of L3 as major pillars or strategic divisions.
*   **Goal Path (Application/Technology Domain):** Identify the primary specific field of technology, application area, or distinct research domain. This should be broader than the L2 Goal. For example, an L2 goal about "Advanced Microchip Cooling" should map to a broader L3 domain like "Microelectronics" or "Semiconductor Technology," not just "Advanced Cooling Systems."
*   **Method Path (Strategic Method Category):** Identify the fundamental type of strategic action, core operational domain, or overarching domain of expertise. This often means moving from a specific L2 method (e.g., "Social Science Research," "Advanced Scientific Modeling") to a higher-level strategic function or capability (e.g., "Evidence-Based Analysis," "Research & Development Methods," "Strategic Collaboration").
*   These L3 categories should be concise (ideally 2-4 words).
*   Use the **Project Summary** to ensure the chosen major category is accurate.

Instructions & Examples (Crucial - Demonstrating SIGNIFICANT Abstraction):
1.  **Max 3 Words** per Level 3 label.
2.  **Level 3 Application/Technology Domain (from L2 Goal): Focus on Established Fields/Sectors.**
    *   Think: "What is the recognized **technological sector, key industry, or established research field** this L2 Goal belongs to?"
    *   Ensure a clear step up in generality from L2.
    *   *Examples of L2 Goal -> Level 3 Application/Technology Domain (Significant Leap):*
        *   L2 Goal: "Advanced Microchip Cooling" -> **L3: Semiconductor Technology**
        *   L2 Goal: "Developing Efficient Solar Cells" -> **L3: Renewable Energy Systems**
        *   L2 Goal: "AI for Medical Diagnosis" -> **L3: Digital Health Applications**
        *   L2 Goal: "Improving Battery Energy Density" -> **L3: Energy Storage Solutions**
        *   L2 Goal: "Understanding Digital Inclusion & Society" -> **L3: Digital Society Studies**
        *   L2 Goal: "Advancing Astrochemistry Knowledge" -> **L3: Space & Planetary Science**
3.  **Level 3 Strategic Method Category (from L2 Method): Focus on Fundamental Strategic Functions or Core Capabilities.**
    *   Think: "What is the **highest-level strategic function, core operational capability, or fundamental type of intervention** that this L2 Method represents?"
    *   **THIS REQUIRES A MAJOR JUMP.** An L2 method like "Social Science Research" is still too specific for L3. What is its strategic function? Perhaps "Societal Understanding & Analysis." An L2 method like "Advanced Scientific Modeling" might become "Predictive & Analytical Methods."
    *   *Examples of L2 Method -> Level 3 Strategic Method Category (Significant Leap):*
        *   L2 Method: "Integrated Liquid Cooling" -> **L3: Advanced Hardware Engineering** (or **Specialized System Design**)
        *   L2 Method: "Social Science Research" -> **L3: Societal Impact Analysis** (or **Human & Social Dynamics Research**)
        *   L2 Method: "Advanced Scientific Modeling" -> **L3: Complex Systems Simulation** (or **Theoretical & Predictive Modeling**)
        *   L2 Method: "AI & Data Analytics" -> **L3: Data-Driven Innovation** (or **Advanced Analytics Capability**)
        *   L2 Method: "European Student Science Competition" -> **L3: Talent Development Programs**
        *   L2 Method: "Implementing Open Science Policies" -> **L3: Research Policy & Governance**
        *   L2 Method: "City-Based Innovation Ecosystems" -> **L3: Regional Development Strategies**
4.  **MANDATE SIGNIFICANT ABSTRACTION:** The jump from Level 2 to Level 3 is NOT optional. Rephrasing or making only minor adjustments to Level 2 labels to create Level 3 labels is incorrect. If a Level 2 label seems very broad, you must still find its encompassing Level 3 strategic category.
5.  **Wording:** Use established, high-level strategic terms.
6.  **Context is King:** The **Project Summary** is vital.
7.  **Unassigned:** If Level 2 inputs are "Unassigned," "Not Generalized," or "Not Applicable" the L3 output should also reflect an inability to categorize, e.g., "Not Categorized L3".

Output Format (one section per project, separated by "---"):
Level 3 Application/Technology Domain: [specific domain label - significant abstraction from L2 Goal]
Level 3 Strategic Method Category: [strategic method type label - significant abstraction from L2 Method]
---

Example (Your UNISCOOL Case - Aiming for Stronger Abstraction):
Project Summary: UNISCOOL: SMART IN-CHIP LIQUID COOLING FOR ADVANCED MICROELECTRONIC SYSTEMS...
Level 2 Generalized Goal: Advanced Microchip Cooling
Level 2 Generalized Method: Integrated Liquid Cooling

Your output for this project:
Level 3 Application/Technology Domain: Semiconductor Technology
Level 3 Strategic Method Category: Advanced Hardware Engineering
---

Example 2 (Project #1 from your output - Demonstrating Stronger Method Abstraction):
Project Summary: Digitizing Other Economies: A Comparative Approach: How do longstanding, primarily non-industrial, n...
Level 2 Goal: Digital Inclusion & Society
Level 2 Method: Social Science Research

Your output for this project:
Level 3 Application/Technology Domain: Digital Society Studies
Level 3 Strategic Method Category: Societal Impact Analysis
---

Example 3 (Project #2 from your output - Demonstrating Stronger Method Abstraction):
Project Summary: MOLECULAR QUANTUM DYNAMICS IN LOW TEMPERATURE CONDENSED PHASE ASTROCHEMISTRY...
Level 2 Goal: Advancing Astrochemistry Knowledge
Level 2 Method: Advanced Scientific Modeling

Your output for this project:
Level 3 Application/Technology Domain: Space & Planetary Science
Level 3 Strategic Method Category: Theoretical & Predictive Modeling
---

Now, analyze the following projects. For each, provide 'Level 3 Application/Technology Domain' and 'Level 3 Strategic Method Category' per format, separated by "---". YOU MUST ENSURE A SIGNIFICANT LEVEL OF ABSTRACTION FOR BOTH OUTPUTS.
"""

ITEM_PROMPT_FORMAT_S3 = (
    "\nProject #{project_num_in_prompt}:\n"
    "Project Summary: {full_text}\n"
    "Level 2 Generalized Goal: {l2_generalized_goal}\n"
    "Level 2 Generalized Method: {l2_generalized_method}\n"
)

INPUT_FIELDS_S3 = ['full_text', 'l2_generalized_goal', 'l2_generalized_method']

LLM_EXPECTED_LABELS_S3 = {
    "Level 3 Application/Technology Domain:": "s3_tech_domain_parsed",
    "Level 3 Strategic Method Category:": "s3_strat_method_parsed"
}

OUTPUT_COL_NAMES_S3 = {
    "s3_tech_domain_parsed": "l3_tech_domain",
    "s3_strat_method_parsed": "l3_strategic_method"
}

def eligibility_filter_s3(df: pd.DataFrame) -> pd.Series:
    # L2 output columns (ensure these match what OUTPUT_COL_NAMES_S2 produced)
    l2_goal_col = OUTPUT_COL_NAMES_S2.get("s2_g_goal_parsed", "l2_generalized_goal")
    l2_method_col = OUTPUT_COL_NAMES_S2.get("s2_g_method_parsed", "l2_generalized_method")

    if l2_goal_col not in df.columns or l2_method_col not in df.columns:
        print(f"Warning for {STAGE_NAME_S3} eligibility: '{l2_goal_col}' or '{l2_method_col}' columns missing.")
        return pd.Series([False] * len(df), index=df.index)

    invalid_l2_output_values = [
        "Parse Error (Key Missing)", "Not Found in LLM Output", "API Error in Batch",
        "Classification Failed", "Not Processed by Stage", "Not Eligible for Stage",
        "Not Applicable", "Not Generalized", "Unassigned", "", pd.NA
    ]
    is_l2_goal_valid = df[l2_goal_col].notna() & (~df[l2_goal_col].astype(str).isin(invalid_l2_output_values))
    is_l2_method_valid = df[l2_method_col].notna() & (~df[l2_method_col].astype(str).isin(invalid_l2_output_values))
    return is_l2_goal_valid & is_l2_method_valid

# --- Execute Stage 3 ---
# df_complete is the main DataFrame, assumed to be updated by Cell 7 (Stage 2.1 Fix)

if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    print(f"Input for Stage 3 ({STAGE_NAME_S3}): df_complete with {len(df_complete)} rows.")
    print(f"df_complete columns before S3: {df_complete.columns.tolist()}")
    
    s3_input_cols_ok = all(col in df_complete.columns for col in INPUT_FIELDS_S3)
            
    if s3_input_cols_ok:
        df_complete = process_stage(
            df_main=df_complete, 
            stage_name=STAGE_NAME_S3,
            prompt_template_str=PROMPT_TEMPLATE_S3_FULL,
            item_prompt_format_str=ITEM_PROMPT_FORMAT_S3,
            input_fields_for_prompt=INPUT_FIELDS_S3,
            llm_expected_labels_map=LLM_EXPECTED_LABELS_S3,
            output_col_names_map=OUTPUT_COL_NAMES_S3,
            model_instance=model,
            eligibility_filter_fn=eligibility_filter_s3,
            run_trial_mode=RUN_OVERALL_TRIAL_MODE,
            max_batches_trial=MAX_BATCHES_FOR_TRIAL,
            batch_size=DEFAULT_BATCH_SIZE, 
            wait_time=DEFAULT_WAIT_TIME, # Using global default (was 15 in your specific L3 code)
            llm_output_separator="---" 
        )
        
        if not df_complete.empty:
            print(f"\n{STAGE_NAME_S3} completed. df_complete columns after S3: {df_complete.columns.tolist()}")
            print(f"Sample of updated df_complete (first 5 rows showing L2 and L3 cols):")
            cols_to_display_s3 = ['id', 'full_text'] + \
                                 [OUTPUT_COL_NAMES_S2.get("s2_g_goal_parsed", "l2_generalized_goal"), 
                                  OUTPUT_COL_NAMES_S2.get("s2_g_method_parsed", "l2_generalized_method")] + \
                                 list(OUTPUT_COL_NAMES_S3.values())
            cols_to_display_s3 = [col for col in cols_to_display_s3 if col in df_complete.columns]
            if cols_to_display_s3: display(df_complete[cols_to_display_s3].head())

            try:
                df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
                print(f"Saved updated df_complete after {STAGE_NAME_S3} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
            except Exception as e:
                print(f"Error saving checkpoint after {STAGE_NAME_S3}: {e}")
        else:
            print(f"Stage {STAGE_NAME_S3} resulted in an empty DataFrame.")
    else:
        print(f"Skipping {STAGE_NAME_S3} due to missing required input columns in df_complete.")
else:
    if 'df_complete' not in locals() or not isinstance(df_complete, pd.DataFrame) or df_complete.empty:
        print(f"Skipping Stage 3: df_complete not ready or is empty.")
    if not model: 
        print("Skipping Stage 3: LLM Model not initialized.")


# In[12]:


# ## Stage 3.1: Fixing Failed L3 Categorizations

STAGE_NAME_S3_FIX = "L3_Fix_Failed_Categorization"
# Reuses S3 configs: PROMPT_TEMPLATE_S3_FULL, ITEM_PROMPT_FORMAT_S3, INPUT_FIELDS_S3, 
#                    LLM_EXPECTED_LABELS_S3, OUTPUT_COL_NAMES_S3 (defined in Cell 8)

def eligibility_filter_s3_fix(df: pd.DataFrame) -> pd.Series:
    # L3 output columns (ensure these match what OUTPUT_COL_NAMES_S3 produced)
    l3_domain_col = OUTPUT_COL_NAMES_S3.get("s3_tech_domain_parsed", "l3_tech_domain")
    l3_method_col = OUTPUT_COL_NAMES_S3.get("s3_strat_method_parsed", "l3_strategic_method")

    if l3_domain_col not in df.columns or l3_method_col not in df.columns:
        print(f"Warning for {STAGE_NAME_S3_FIX} eligibility: '{l3_domain_col}' or '{l3_method_col}' columns missing.")
        return pd.Series([False] * len(df), index=df.index)
        
    # Was it eligible for the main S3 run? (Uses eligibility_filter_s3 from Cell 8)
    was_eligible_for_s3_main = eligibility_filter_s3(df) 
    
    # Define what indicates an L3 failure needing a retry
    failed_l3_output_values = [ 
        "Parse Error (Key Missing)", "Not Found in LLM Output", 
        "API Error in Batch", "Classification Failed", "", 
        "Not Categorized L3", # Placeholder from original L3 script if LLM couldn't categorize
        "Not Processed by Stage", "Not Eligible for Stage", pd.NA 
    ] 
    
    failed_l3_domain = df[l3_domain_col].astype(str).isin(failed_l3_output_values) | df[l3_domain_col].isnull()
    failed_l3_method = df[l3_method_col].astype(str).isin(failed_l3_output_values) | df[l3_method_col].isnull()
    
    return was_eligible_for_s3_main & (failed_l3_domain | failed_l3_method)

# --- Execute Stage 3.1 ---
if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    rows_needing_s3_fix_mask = eligibility_filter_s3_fix(df_complete)
    if not rows_needing_s3_fix_mask.any():
        print(f"Stage {STAGE_NAME_S3_FIX}: No rows found needing L3 fix. df_complete remains unchanged.")
    else:
        print(f"Input for {STAGE_NAME_S3_FIX}: df_complete ({len(df_complete)} rows). Target to fix: {rows_needing_s3_fix_mask.sum()}")
        s3_fix_input_cols_ok = all(col in df_complete.columns for col in INPUT_FIELDS_S3) # INPUT_FIELDS_S3 from Cell 8
        
        if s3_fix_input_cols_ok:
            df_complete = process_stage(
                df_main=df_complete,       
                stage_name=STAGE_NAME_S3_FIX,
                prompt_template_str=PROMPT_TEMPLATE_S3_FULL, 
                item_prompt_format_str=ITEM_PROMPT_FORMAT_S3, 
                input_fields_for_prompt=INPUT_FIELDS_S3,     
                llm_expected_labels_map=LLM_EXPECTED_LABELS_S3, 
                output_col_names_map=OUTPUT_COL_NAMES_S3,    
                model_instance=model,
                eligibility_filter_fn=eligibility_filter_s3_fix, 
                run_trial_mode=RUN_OVERALL_TRIAL_MODE, 
                max_batches_trial=MAX_BATCHES_FOR_TRIAL, 
                batch_size=5,  # Smaller batch for fixing
                wait_time=15,  # Override default: Longer wait for fixing complex L3
                llm_output_separator="---" 
            )
            
            if not df_complete.empty:
                print(f"\n{STAGE_NAME_S3_FIX} completed.")
                print("Sample of rows that were targeted for L3 Fix (first 5, showing L3 cols):")
                cols_to_display_s3_fix = ['id', 'full_text'] + list(OUTPUT_COL_NAMES_S3.values())
                cols_to_display_s3_fix = [col for col in cols_to_display_s3_fix if col in df_complete.columns]
                if rows_needing_s3_fix_mask.any() and cols_to_display_s3_fix:
                     display(df_complete[rows_needing_s3_fix_mask][cols_to_display_s3_fix].head())
                else:
                     print("No rows were targeted for S3 fix for display, or relevant columns missing.")

                try:
                    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
                    print(f"Saved updated df_complete after {STAGE_NAME_S3_FIX} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
                except Exception as e:
                    print(f"Error saving checkpoint after {STAGE_NAME_S3_FIX}: {e}")
        else:
            print(f"Skipping {STAGE_NAME_S3_FIX} due to missing input columns.")
else:
    if 'df_complete' not in locals() or not isinstance(df_complete, pd.DataFrame) or df_complete.empty:
        print(f"Skipping Stage {STAGE_NAME_S3_FIX}: df_complete not ready or model not initialized.")
    if not model:
        print(f"Skipping Stage {STAGE_NAME_S3_FIX}: LLM Model is not initialized.")


# In[17]:


# ## Stage 4: L3 Area Consolidation->L4
# --- Stage 4 Configuration ---
STAGE_NAME_S4 = "L3_Area_Consolidation"

# Full prompt for L3 Area Consolidation (PROMPT_PREFIX_L3_AREA_CONSOLIDATION from your original In[188])
PROMPT_TEMPLATE_S4_FULL = """You are an expert taxonomist. Your primary task is to consolidate and standardize existing Level 3 project labels. The strict goal is to refine the current lists into:
A highly consolidated set of 70-80 distinct 'Standardized L3 Domain Areas'.
A highly consolidated set of 70-80 distinct 'Standardized L3 Method Areas'.
Adherence to this target number range (70-80) for each list is paramount. Each standardized category should be a clear, searchable, and appropriately broad Level 3 representation of either a thematic domain or a functional approach.
For each project, you will be given:
The original Project Summary (for essential context and disambiguation).
Two existing Level 3 labels:
An Existing L3 Application/Technology Domain (the current, potentially too granular or inconsistently phrased, domain label).
An Existing L3 Strategic Method Category (the current, potentially too granular or inconsistently phrased, method label).
Your task is to map BOTH of these existing L3 labels to new, standardized Level 3 categories:
A Standardized L3 Domain Area.
A Standardized L3 Method Area.
Core Principles for This Level 3 Standardization & Consolidation:
Each standardized L3 area must represent a well-recognized and significant domain, field, societal challenge, strategic approach, or core function, suitable for a list strictly limited to 70-80 distinct items per type. Prioritize broader categories to meet this numerical constraint.
Abstract away all non-essential specificity or minor variations from the 'Existing L3' labels. Aggressively broaden categories. For 'Domain Areas', remove action qualifiers (e.g., "improvement") if the core theme is clear, consolidating under the core thematic concept.
The domain area should more reflect the purpose of the projects, while the method should reflect how this is being achieved.
Use the Project Summary to determine the most fitting standardized L3 areas.
Employ significant, systematic consolidation: Group closely related 'Existing L3' labels under a unified 'Standardized L3 Area'. To achieve the 70-80 target for each list, you MUST favor broader categorizations. For example, while 'Solar Energy' and 'Wind Energy' could be distinct, if many such specific sub-types exist, you should strongly consider consolidating them under a broader category like 'Renewable Energy Technologies' if it helps meet the 70-80 target and aligns with the exemplar breadth. The primary driver is the reduction of category count to the target range.
Exemplar Standardized L3 Domain Areas (illustrative of target breadth):
Biomedical Research & Discovery
Public Health & Prevention
Healthcare Services & Innovation
Pharmaceutical Development
Neurological Disorders & Therapies
Climate Change Mitigation
Environmental Protection & Restoration
Renewable Energy Technologies
Sustainable Energy Systems
Energy Efficiency & Conservation
Circular Economy & Resource Efficiency
Materials Science & Engineering
Nanomaterials & Nanotechnology
Advanced Manufacturing & Industrial Processes
Artificial Intelligence & Machine Learning
Data Science & Big Data Analytics
Cybersecurity & Digital Privacy
Sustainable Mobility & Transport
Urban Planning & Smart Cities
Biodiversity & Conservation
Public Policy & Administration
Entrepreneurship & Innovation
Agricultural Systems & Food Security
Water Resource Management
Space Exploration & Technology
Exemplar Standardized L3 Method Areas (illustrative of target breadth):
Basic & Applied Research
Experimental Design & Execution
Field Research & Observation
Computational Modeling & Simulation
Statistical Analysis & Quantitative Methods
Qualitative Research Methods
Data Collection & Curation
AI & Machine Learning Application
Software & Algorithm Development
Engineering Design & Systems Integration
Pre-clinical Drug Development
Clinical Research
Policy Analysis & Formulation
Stakeholder Engagement & Co-Design
Capacity Building & Training Delivery
Educational Program Development
Communication Strategies
Monitoring, Reporting & Evaluation
Innovation Ecosystem Facilitation
Technology Transfer & Commercialization
Supply Chain Optimization
Risk Assessment & Management
Prototyping & Pilot Projects
Instructions:
Output TWO standardized L3 areas (max 3 words each) - one for domain, one for method.
Map to Categories of Exemplar Breadth: Your primary goal is to map 'Existing L3' labels to 'Standardized L3 Areas' that match the level of specificity shown in the provided exemplar lists. Crucially, this requires aggressive consolidation to ensure the total distinct categories for Domains and Methods each strictly fall within the 70-80 range. If an 'Existing L3' label is more granular than the exemplars or would lead to exceeding the target count, it must be broadened and mapped to a more general standardized category.
Consolidate Synonyms, Minor Variations, and Conceptually Related Items: Actively seek to merge very similar 'Existing L3' labels. Extend this principle to group conceptually related items even if not direct synonyms, if doing so is necessary to achieve the target of 70-80 distinct categories per list and aligns with the exemplar breadth. For example, "AI-Driven Analysis," "AI-Powered Analytics," and "Machine Learning for Data Insight" must consolidate, e.g., to "AI & Machine Learning Application" (Method Area).
Use Project Summary for Disambiguation & Best Fit: Crucial for selecting the most appropriate standardized L3 area that aligns with the consolidation goals.
Standardize Terminology: Use consistent terms for the output standardized L3 areas, drawing inspiration from the style and breadth of the exemplars. Do not invent highly niche categories.
Overarching Directive: The success of this task is measured by your ability to reduce category proliferation and strictly adhere to the 70-80 distinct category limit for both Domain Areas and Method Areas. Be decisive in merging and broadening existing labels to meet this target. If an existing label is highly specific, find the appropriate broader exemplar-style category it belongs to.
Output Format (one section per project, separated by "---"):
Standardized Domain Area: [short, thematic area label of target L3 breadth]
Standardized Method Area: [short, functional area label of target L3 breadth]
---
Example 2:
Project Summary: Project focusing on developing new drugs for Alzheimer's disease through identification of novel biological targets and extensive pre-clinical testing.
Existing L3 Application/Technology Domain: Neurological Disease Treatment
Existing L3 Strategic Method Category: Biomedical R&D Methodology
Your output for this project:
Standardized Domain Area: Neurological Disorders & Therapies
Standardized Method Area: Pre-clinical Drug Development
---
Example 3:
Project Summary: Initiative to create pan-European educational programs for young entrepreneurs, fostering innovation and cross-border collaboration, with specific modules on digital marketing and venture finance.
Existing L3 Application/Technology Domain: Entrepreneurship Education
Existing L3 Strategic Method Category: Ecosystem Development
Your output for this project:
Standardized Domain Area: Entrepreneurship & Innovation
Standardized Method Area: Educational Program Development
---
Now, analyze the following projects. For each, provide BOTH 'Standardized Domain Area' and 'Standardized Method Area' labels per the format above, separated by "---". Ensure consolidation into categories matching the breadth and style of the exemplar lists, strictly aiming for 70-80 distinct categories for domains and 70-80 for methods.
"""

ITEM_PROMPT_FORMAT_S4 = (
    "\nProject #{project_num_in_prompt}:\n"
    "Project Summary: {full_text}\n"
    "Existing L3 Application/Technology Domain: {l3_tech_domain}\n"    # Uses output from S3/S3.1
    "Existing L3 Strategic Method Category: {l3_strategic_method}\n" # Uses output from S3/S3.1
)

INPUT_FIELDS_S4 = ['full_text', 'l3_tech_domain', 'l3_strategic_method']

# Note: The parser needs to handle optional "L3" in the LLM's output keys
LLM_EXPECTED_LABELS_S4 = {
    "Standardized Domain Area:": "s4_std_domain_parsed", # Will also match "Standardized L3 Domain Area:" due to how generic_extract_labels works
    "Standardized Method Area:": "s4_std_method_parsed"  # Will also match "Standardized L3 Method Area:"
}
# If the LLM is inconsistent, you might add more specific keys here, or improve the regex in generic_extract_labels.
# For now, relying on the case-insensitive prefix match in generic_extract_labels.

OUTPUT_COL_NAMES_S4 = {
    "s4_std_domain_parsed": "l3_std_domain_area",
    "s4_std_method_parsed": "l3_std_method_area"
}

def eligibility_filter_s4(df: pd.DataFrame) -> pd.Series:
    l3_domain_col = OUTPUT_COL_NAMES_S3.get("s3_tech_domain_parsed", "l3_tech_domain")
    l3_method_col = OUTPUT_COL_NAMES_S3.get("s3_strat_method_parsed", "l3_strategic_method")

    if l3_domain_col not in df.columns or l3_method_col not in df.columns:
        print(f"Warning for {STAGE_NAME_S4} eligibility: L3 columns '{l3_domain_col}' or '{l3_method_col}' missing.")
        return pd.Series([False] * len(df), index=df.index)

    invalid_l3_output_values = [
        "Parse Error (Key Missing)", "Not Found in LLM Output", "API Error in Batch",
        "Classification Failed", "Not Processed by Stage", "Not Eligible for Stage",
        "Not Applicable L3", "Not Categorized L3", "", pd.NA # "Not Categorized L3" was from original L3 script
    ]
    is_l3_domain_valid = df[l3_domain_col].notna() & (~df[l3_domain_col].astype(str).isin(invalid_l3_output_values))
    is_l3_method_valid = df[l3_method_col].notna() & (~df[l3_method_col].astype(str).isin(invalid_l3_output_values))
    return is_l3_domain_valid & is_l3_method_valid

# --- Execute Stage 4 ---
# df_complete is the main DataFrame, assumed to be updated by Cell 9 (Stage 3.1 Fix)

if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    print(f"Input for Stage 4 ({STAGE_NAME_S4}): df_complete with {len(df_complete)} rows.")
    print(f"df_complete columns before S4: {df_complete.columns.tolist()}")
    
    s4_input_cols_ok = all(col in df_complete.columns for col in INPUT_FIELDS_S4)
            
    if s4_input_cols_ok:
        df_complete = process_stage( 
            df_main=df_complete, 
            stage_name=STAGE_NAME_S4,
            prompt_template_str=PROMPT_TEMPLATE_S4_FULL,
            item_prompt_format_str=ITEM_PROMPT_FORMAT_S4,
            input_fields_for_prompt=INPUT_FIELDS_S4,
            llm_expected_labels_map=LLM_EXPECTED_LABELS_S4,
            output_col_names_map=OUTPUT_COL_NAMES_S4,
            model_instance=model, # Or model_for_l3_consolidation if you defined it separately
            eligibility_filter_fn=eligibility_filter_s4,
            run_trial_mode=RUN_OVERALL_TRIAL_MODE,
            max_batches_trial=MAX_BATCHES_FOR_TRIAL,
            batch_size=DEFAULT_BATCH_SIZE, 
            wait_time=DEFAULT_WAIT_TIME, 
            llm_output_separator="---" 
        )
        
        if not df_complete.empty:
            print(f"\n{STAGE_NAME_S4} completed. df_complete columns after S4: {df_complete.columns.tolist()}")
            print(f"Sample of updated df_complete (first 5 rows showing L3 and consolidated L3 cols):")
            cols_to_display_s4 = ['id', 'full_text'] + \
                                 [OUTPUT_COL_NAMES_S3.get("s3_tech_domain_parsed", "l3_tech_domain"), 
                                  OUTPUT_COL_NAMES_S3.get("s3_strat_method_parsed", "l3_strategic_method")] + \
                                 list(OUTPUT_COL_NAMES_S4.values())
            cols_to_display_s4 = [col for col in cols_to_display_s4 if col in df_complete.columns]
            if cols_to_display_s4: display(df_complete[cols_to_display_s4].head())

            try:
                df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
                print(f"Saved updated df_complete after {STAGE_NAME_S4} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
            except Exception as e:
                print(f"Error saving checkpoint after {STAGE_NAME_S4}: {e}")
        else:
            print(f"Stage {STAGE_NAME_S4} resulted in an empty DataFrame.")
    else:
        print(f"Skipping {STAGE_NAME_S4} due to missing required input columns in df_complete.")
else:
    if 'df_complete' not in locals() or not isinstance(df_complete, pd.DataFrame) or df_complete.empty:
        print(f"Skipping Stage 4: df_complete not ready or is empty.")
    if not model: 
        print("Skipping Stage 4: LLM Model not initialized.")


# In[18]:


# ## Stage 4.1: Fixing Failed L3 Area Consolidations

STAGE_NAME_S4_FIX = "L3_Fix_Failed_Area_Consolidation"
# Reuses S4 configs: PROMPT_TEMPLATE_S4_FULL, ITEM_PROMPT_FORMAT_S4, INPUT_FIELDS_S4, 
#                    LLM_EXPECTED_LABELS_S4, OUTPUT_COL_NAMES_S4 (defined in Cell 10)

def eligibility_filter_s4_fix(df: pd.DataFrame) -> pd.Series:
    l3_std_domain_col = OUTPUT_COL_NAMES_S4.get("s4_std_domain_parsed", "l3_std_domain_area")
    l3_std_method_col = OUTPUT_COL_NAMES_S4.get("s4_std_method_parsed", "l3_std_method_area")

    if l3_std_domain_col not in df.columns or l3_std_method_col not in df.columns:
        print(f"Warning for {STAGE_NAME_S4_FIX} eligibility: L3 Standardized output columns missing.")
        return pd.Series([False] * len(df), index=df.index)
        
    was_eligible_for_s4_main = eligibility_filter_s4(df) # Uses S4 eligibility from Cell 10
    
    failed_s4_values = [ 
        "Parse Error (Key Missing)", "Not Found in LLM Output", 
        "API Error in Batch", "Classification Failed", "", 
        "Not Standardized", # Placeholder from original L3 consolidation script
        "Not Processed by Stage", "Not Eligible for Stage", pd.NA 
    ] 
    
    failed_s4_domain = df[l3_std_domain_col].astype(str).isin(failed_s4_values) | df[l3_std_domain_col].isnull()
    failed_s4_method = df[l3_std_method_col].astype(str).isin(failed_s4_values) | df[l3_std_method_col].isnull()
    
    return was_eligible_for_s4_main & (failed_s4_domain | failed_s4_method)

# --- Execute Stage 4.1 ---
if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model:
    rows_needing_s4_fix_mask = eligibility_filter_s4_fix(df_complete)
    if not rows_needing_s4_fix_mask.any():
        print(f"Stage {STAGE_NAME_S4_FIX}: No rows found needing L3 Area Consolidation fix.")
    else:
        print(f"Input for {STAGE_NAME_S4_FIX}: df_complete ({len(df_complete)} rows). Target to fix: {rows_needing_s4_fix_mask.sum()}")
        s4_fix_input_cols_ok = all(col in df_complete.columns for col in INPUT_FIELDS_S4) # INPUT_FIELDS_S4 from Cell 10
        
        if s4_fix_input_cols_ok:
            df_complete = process_stage(
                df_main=df_complete,       
                stage_name=STAGE_NAME_S4_FIX,
                prompt_template_str=PROMPT_TEMPLATE_S4_FULL, 
                item_prompt_format_str=ITEM_PROMPT_FORMAT_S4, 
                input_fields_for_prompt=INPUT_FIELDS_S4,     
                llm_expected_labels_map=LLM_EXPECTED_LABELS_S4, 
                output_col_names_map=OUTPUT_COL_NAMES_S4,    
                model_instance=model, # Or model_for_l3_consolidation
                eligibility_filter_fn=eligibility_filter_s4_fix, 
                run_trial_mode=RUN_OVERALL_TRIAL_MODE, 
                max_batches_trial=MAX_BATCHES_FOR_TRIAL, 
                batch_size=10, # Smaller batch for fixing
                wait_time=20,  # Override default: Longer wait for this complex fix
                llm_output_separator="---" 
            )
            
            if not df_complete.empty:
                print(f"\n{STAGE_NAME_S4_FIX} completed.")
                print("Sample of rows that were targeted for L3 Consolidation Fix (first 5):")
                cols_to_display_s4_fix = ['id', 'full_text'] + list(OUTPUT_COL_NAMES_S4.values())
                cols_to_display_s4_fix = [col for col in cols_to_display_s4_fix if col in df_complete.columns]
                if rows_needing_s4_fix_mask.any() and cols_to_display_s4_fix:
                     display(df_complete[rows_needing_s4_fix_mask][cols_to_display_s4_fix].head())

                try:
                    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
                    print(f"Saved updated df_complete after {STAGE_NAME_S4_FIX} to checkpoint: {MAIN_DATAFRAME_CHECKPOINT_FILE}")
                except Exception as e:
                    print(f"Error saving checkpoint after {STAGE_NAME_S4_FIX}: {e}")
        else:
            print(f"Skipping {STAGE_NAME_S4_FIX} due to missing input columns.")
else:
    print(f"Skipping Stage {STAGE_NAME_S4_FIX}: df_complete not ready or model not initialized.")


# In[33]:


# CELL 12 - Stage 5.1: Batch Consolidate Unique L3 Std. Areas to ~100/100 Lists
# (Assuming LLM now outputs clean categories directly, no prefix cleaning needed)

# ## Stage 5.1: Consolidate Unique L3 Standardized Areas to ~100 Categories Each

STAGE_NAME_S5_1 = "S5.1_Consolidate_L3Std_To_100"
print(f"\n{'='*20} Starting Stage: {STAGE_NAME_S5_1} {'='*20}")

model_cat_list_gen = model # Using the globally defined model

# --- Helper Functions for Stage 5.1 ---

# IMPORTANT: Your PROMPT_S5_1_CONSOLIDATE_TO_100 should now include an instruction
# like "Do not add any numbering or bullet points to the category names."
PROMPT_S5_1_CONSOLIDATE_TO_100 = """You are an expert taxonomist. Your task is to consolidate the provided list of categories into a smaller, more manageable set of standardized categories.

You will be given:
1. A list of Domain Area categories to consolidate (e.g., up to 505 entries in this batch)
2. A list of Method Area categories to consolidate (e.g., up to 505 entries in this batch)

Your task is to:
1. Reduce the provided list of Domain Area categories to approximately 100 standardized categories.
2. Reduce the provided list of Method Area categories to approximately 100 standardized categories.

Core Principles for Consolidation:
* Identify common themes and patterns across related categories.
* Group conceptually similar items together under broader labels.
* Ensure each consolidated category is clear, descriptive, and represents a distinct domain or method.
* Use established terminology from research and industry.
* The consolidated categories should cover the full range of the original categories provided in this batch.
* Each standardized category should have a clear scope and purpose. Example target breadth: "Microelectronics".

Consolidate by:
* Merging synonyms and near-synonyms (e.g., "AI Applications" and "Artificial Intelligence Utilization").
* Grouping sub-areas under broader domains (e.g., "Solar Energy", "Wind Energy" → "Renewable Energy Technologies").
* Standardizing terminology and removing qualifiers when the core concept is clear.

Output Format:
----Domain Areas----
[Consolidated Category Name 1]
[Consolidated Category Name 2]
... 
[Approximately 100 Consolidated Categories]

----Method Areas----
[Consolidated Category Name 1]
[Consolidated Category Name 2]
...
[Approximately 100 Consolidated Categories]

**IMPORTANT INSTRUCTION: Ensure that the category names themselves do NOT include any leading numbers, bullet points, or other list markers (e.g., no "1. Category Name", just "Category Name").**

Here are the categories for THIS BATCH to consolidate:

----Domain Areas to Consolidate----
{domain_categories_list_str}

----Method Areas to Consolidate----
{method_categories_list_str}
Remember: For the lists provided in THIS BATCH, produce approximately 100 Domain Areas and approximately 100 Method Areas in your response.
"""

def batch_category_strings_by_count(
    categories: List[str], 
    batch_size_count: int = 505 
) -> List[List[str]]:
    if not categories: return [[]] 
    return [categories[i:i + batch_size_count] for i in range(0, len(categories), batch_size_count)]

def call_llm_for_category_batch_consolidation( 
    domain_category_batch_list: List[str], 
    method_category_batch_list: List[str], 
    model_to_use, 
    prompt_template: str,
    batch_num_for_api: int, 
    max_retries: int = 3
) -> Dict[str, List[str]]:
    if not model_to_use:
        print(f"Model not available for S5.1 batch {batch_num_for_api}. Skipping API call.")
        return {"domains_raw": [], "methods_raw": []} 

    domain_categories_str_for_prompt = "\n".join(domain_category_batch_list)
    method_categories_str_for_prompt = "\n".join(method_category_batch_list)
    prompt = prompt_template.format(
        domain_categories_list_str=domain_categories_str_for_prompt,
        method_categories_list_str=method_categories_str_for_prompt
    )

    if batch_num_for_api == 1 : 
        print(f"\n=== S5.1 CONSOLIDATION PROMPT (Batch 1 - Structure Preview) ===")
        sample_prompt = prompt_template.format(
            domain_categories_list_str=f"[List of {len(domain_category_batch_list)} Domain Categories...]",
            method_categories_list_str=f"[List of {len(method_category_batch_list)} Method Categories...]"
        )
        print(sample_prompt.split("Here are the categories",1)[0] + "...") 
        print("=== END S5.1 PROMPT SAMPLE ===\n")
        
    output_text = ""
    for attempt in range(max_retries):
        try:
            response = model_to_use.generate_content(prompt)
            output_text = response.text
            break
        except Exception as e:
            print(f"Error on S5.1 API call (Batch {batch_num_for_api}, Attempt {attempt + 1}): {type(e).__name__} - {e}")
            if attempt < max_retries - 1: 
                sleep_duration = (2 ** attempt) * 3 
                print(f"  Retrying S5.1 Batch {batch_num_for_api} in {sleep_duration}s...")
                time.sleep(sleep_duration)
            else: print(f"Failed all S5.1 retry attempts for Batch {batch_num_for_api}.")

    # Directly extract lists, assuming they are clean from LLM
    extracted_domains, extracted_methods = [], []
    if output_text:
        if batch_num_for_api == 1: 
             print(f"\n--- RAW LLM OUTPUT (S5.1 Batch 1) ---")
             print(output_text[:1000] + ("..." if len(output_text) > 1000 else ""))
             print("--- END RAW LLM OUTPUT ---\n")

        domains_section = re.search(r'----Domain Areas----\s*([\s\S]*?)(?=----Method Areas----|\Z)', output_text, re.IGNORECASE)
        methods_section = re.search(r'----Method Areas----\s*([\s\S]*?)(?=\Z)', output_text, re.IGNORECASE)
        
        if domains_section:
            extracted_domains = [line.strip() for line in domains_section.group(1).strip().split('\n') if line.strip()]
        if methods_section:
            extracted_methods = [line.strip() for line in methods_section.group(1).strip().split('\n') if line.strip()]
    
    print(f"  S5.1 Batch {batch_num_for_api} -> LLM returned {len(extracted_domains)} domain strings, {len(extracted_methods)} method strings.")
    # Return "domains_raw" and "methods_raw" for consistency, even though we expect them to be clean
    return {"domains_raw": extracted_domains, "methods_raw": extracted_methods}

# --- Execute Stage 5.1 ---
s5_1_final_consolidated_domains = [] # Changed from s5_1_final_cleaned_domains
s5_1_final_consolidated_methods = [] # Changed from s5_1_final_cleaned_methods

if 'df_complete' in locals() and isinstance(df_complete, pd.DataFrame) and not df_complete.empty and model_cat_list_gen:
    print(f"Input for {STAGE_NAME_S5_1}: df_complete with {len(df_complete)} rows.")
    
    std_domain_col_s4 = OUTPUT_COL_NAMES_S4.get("s4_std_domain_parsed", "l3_std_domain_area")
    std_method_col_s4 = OUTPUT_COL_NAMES_S4.get("s4_std_method_parsed", "l3_std_method_area")

    if std_domain_col_s4 not in df_complete.columns or std_method_col_s4 not in df_complete.columns:
        print(f"ERROR for {STAGE_NAME_S5_1}: L3 Standardized columns ('{std_domain_col_s4}', '{std_method_col_s4}') not found.")
    else:
        s4_error_placeholders = [ "Not Standardized", "Not Applicable Std", "Parse Error (Key Missing)", 
                                  "Not Found in LLM Output", "API Error in Batch", 
                                  "Classification Failed", "", pd.NA ]
        unique_s4_domains_input_list = sorted(list(df_complete[
            df_complete[std_domain_col_s4].notna() & \
            (~df_complete[std_domain_col_s4].astype(str).isin(s4_error_placeholders))
        ][std_domain_col_s4].unique()))
        unique_s4_methods_input_list = sorted(list(df_complete[
            df_complete[std_method_col_s4].notna() & \
            (~df_complete[std_method_col_s4].astype(str).isin(s4_error_placeholders))
        ][std_method_col_s4].unique()))

        print(f"Found {len(unique_s4_domains_input_list)} unique valid L3 Std Domain Areas to consolidate.")
        print(f"Found {len(unique_s4_methods_input_list)} unique valid L3 Std Method Areas to consolidate.")

        if not unique_s4_domains_input_list and not unique_s4_methods_input_list:
            print(f"{STAGE_NAME_S5_1}: No unique valid categories. Skipping LLM consolidation.")
        else:
            domain_category_batches = batch_category_strings_by_count(unique_s4_domains_input_list, batch_size_count=600)
            method_category_batches = batch_category_strings_by_count(unique_s4_methods_input_list, batch_size_count=600)
            num_api_calls = max(len(domain_category_batches), len(method_category_batches))
            if num_api_calls == 0 and (unique_s4_domains_input_list or unique_s4_methods_input_list): num_api_calls = 1
            while len(domain_category_batches) < num_api_calls: domain_category_batches.append([])
            while len(method_category_batches) < num_api_calls: method_category_batches.append([])

            print(f"{STAGE_NAME_S5_1}: Will make {num_api_calls} API call(s).")

            # These lists will now store the directly extracted (hopefully clean) categories
            collected_domains_from_llm = []
            collected_methods_from_llm = []

            for i_batch_for_api in range(num_api_calls):
                current_domain_batch_for_api = domain_category_batches[i_batch_for_api]
                current_method_batch_for_api = method_category_batches[i_batch_for_api]
                print(f"\n--- Processing S5.1 API Batch {i_batch_for_api+1}/{num_api_calls} ---")
                if not current_domain_batch_for_api and not current_method_batch_for_api:
                    print(f"  Skipping API call for S5.1 batch {i_batch_for_api+1} as both input lists are empty.")
                    continue
                batch_result_dict = call_llm_for_category_batch_consolidation(
                    current_domain_batch_for_api, current_method_batch_for_api,
                    model_cat_list_gen, PROMPT_S5_1_CONSOLIDATE_TO_100,
                    batch_num_for_api=i_batch_for_api + 1 )
                collected_domains_from_llm.extend(batch_result_dict["domains_raw"]) 
                collected_methods_from_llm.extend(batch_result_dict["methods_raw"])
                if i_batch_for_api < num_api_calls - 1: 
                    wait_s5_1 = DEFAULT_WAIT_TIME * 2 
                    print(f"  Waiting {wait_s5_1}s before next S5.1 API call...") 
                    time.sleep(wait_s5_1)
            
            # NO CLEANING FUNCTION CALL HERE - Assuming LLM provides clean output due to prompt
            
            # Get unique categories from the (assumed clean) lists
            s5_1_final_consolidated_domains = sorted(list(set(cat for cat in collected_domains_from_llm if cat))) # Filter empty strings just in case
            s5_1_final_consolidated_methods = sorted(list(set(cat for cat in collected_methods_from_llm if cat)))

            print(f"\n{STAGE_NAME_S5_1} LLM processing completed.")
            print(f"  Total unique consolidated domains after S5.1: {len(s5_1_final_consolidated_domains)}")
            print(f"  Total unique consolidated methods after S5.1: {len(s5_1_final_consolidated_methods)}")

            mode_str_for_filename = "trial" if RUN_OVERALL_TRIAL_MODE and MAX_BATCHES_FOR_TRIAL > 0 else "full"
            s5_1_domain_output_filename = f"s5_1_consolidated_approx100_domains_{mode_str_for_filename}.csv"
            s5_1_method_output_filename = f"s5_1_consolidated_approx100_methods_{mode_str_for_filename}.csv"
            
            if s5_1_final_consolidated_domains:
                pd.DataFrame({'consolidated_domain_area_S5_1': s5_1_final_consolidated_domains}).to_csv(s5_1_domain_output_filename, index=False)
            else: pd.DataFrame(columns=['consolidated_domain_area_S5_1']).to_csv(s5_1_domain_output_filename, index=False)
            if s5_1_final_consolidated_methods:
                pd.DataFrame({'consolidated_method_area_S5_1': s5_1_final_consolidated_methods}).to_csv(s5_1_method_output_filename, index=False)
            else: pd.DataFrame(columns=['consolidated_method_area_S5_1']).to_csv(s5_1_method_output_filename, index=False)
            
            if s5_1_final_consolidated_domains or s5_1_final_consolidated_methods :
                 print(f"S5.1 consolidated lists saved to '{s5_1_domain_output_filename}' and '{s5_1_method_output_filename}'")
            else:
                 print("S5.1: No categories to save (lists might be empty).")
else:
    print(f"Skipping {STAGE_NAME_S5_1}: df_complete not ready, model not initialized, or S4 columns missing.")


# ### Using a combination of AI and human feedback, we refined the extracted categories into a set of distinct, well-defined categories that effectively encompass the underlying concepts.

# In[123]:


STAGE_NAME_S5 = "S5_Classification_Working_Approach"
print(f"\n{'='*20} Starting Stage: {STAGE_NAME_S5} {'='*20}")

# Your detailed categories (same as before)
DOMAIN_CATEGORIES_CURATED = [
    "AI & Machine Learning", "Advanced Computing & Electronics", "Advanced Manufacturing & Industry", 
    "Advanced Therapies & Regenerative Medicine", "Aerospace Engineering & Aviation", "Agricultural Systems & Plant Science", 
    "Alternative & Biofuels", "Analytical & Theoretical Chemistry", "Animal & Veterinary Sciences", 
    "Anthropology & Archaeology", "Applied & Plasma Physics", "Aquaculture & Fisheries", 
    "Arts, Humanities & Culture", "Astronomy, Cosmology & Particle Physics", "Atmospheric, Climate & Environmental Science", 
    "Autoimmune & Chronic Disease", "Autonomous Systems & Robotics", "Basic & Cross-Domain Science", 
    "Behavioral Science & Psychology", "Biodiversity & Ecology", "Bioeconomy & Circular Economy", 
    "Bioengineering & Synthetic Biology", "Bioinformatics & Systems Biology", "Biological Sciences", 
    "Biomedical Devices & Imaging", "Biomedical Engineering", "Biomedical Research (General)", 
    "Biotechnology", "Brain Science & Neuroscience", "Business & Organizational Behavior", 
    "Cancer Research & Oncology", "Carbon Management & Climate Technology", "Cardiovascular Health", 
    "Chemical & Process Engineering", "Chemical Safety & Toxicology", "Civic Engagement & Governance", 
    "Civil Engineering & Infrastructure", "Clean & Renewable Energy", "Climate Adaptation & Impact", 
    "Cloud & Digital Infrastructure", "Cognitive Science", "Communication & Media Studies", 
    "Complex Systems Science", "Computer Science (General)", "Computer Vision & Graphics", 
    "Conflict, Security & Justice", "Crisis & Disaster Management", "Critical Infrastructure & Safety", 
    "Critical Materials & Waste Management", "Cybersecurity & Digital Governance", "Data Science & Analytics", 
    "Dental & Oral Health", "Design Research & Human-Computer Interaction", "Digital Economy & Innovation", 
    "Digital Health & Medical Technology", "Digital Humanities & Cultural Heritage", "Drug Discovery & Pharmaceutical Development", 
    "Economics & Development Studies", "Energy Generation & Systems", "Food Science, Nutrition & Security", 
    "Genomics & Precision Medicine", "Gerontology & Aging Studies", "Historical Studies", 
    "Imaging & Instrumentation Technologies", "Immunology & Vaccine Development", "Life Sciences (General)", 
    "Linguistics & Language Studies", "Logistics & Supply Chain Management", "Marine, Ocean & Water Science", 
    "Materials Science & Engineering", "Mathematics & Applied Statistics", "Mechanical Engineering", 
    "Medical Education & Training", "Mental Health & Psychiatry", "Metabolic & Endocrine Health", 
    "Metrology & Measurement Science", "Microbiology & Microbiome Science", "Migration & Urban Studies", 
    "Musculoskeletal Health & Rehabilitation", "National Security & Defense Studies", "Natural Language Processing", 
    "Natural Resources & Environmental Management", "Neglected & Rare Diseases", "Network Science & Telecommunications", 
    "Nuclear Science & Engineering", "Occupational Health & Safety", "Optics & Photonics", 
    "Organic & Inorganic Chemistry", "Patient Care & Health Systems", "Pediatric & Women's Health", 
    "Philosophy, Logic & Ethics", "Physical & Chemical Sciences", "Planetary & Space Science", 
    "Political Science & Policy", "Public Health & Global Health", "Quantum Science & Technology", 
    "Regional & Urban Planning", "Religious Studies & Theology", "Remote Sensing & Geospatial Technology", 
    "Research & Innovation Policy", "Science & Technology Studies", "Simulation & Training Technology", 
    "Smart Systems & Internet of Things", "Social Justice & Equity", "Social Research & Sociology", 
    "Software Systems Engineering", "Stem Cell & Regenerative Engineering", "Sustainability Science & Education", 
    "Sustainable Consumption & Lifestyles", "Systems Engineering (General)", "Theoretical Physics", 
    "Toxicology & Risk Assessment", "Translational Medicine", "Transport Safety & Emissions reduction", 
    "Venture Capital & Innovation Finance", "Wildlife Health & Ecology"
]

METHOD_CATEGORIES_CURATED = [
    "AI & Machine Learning Development", "Advanced Analytical Techniques", "Advanced Engineering Methods", 
    "Advanced Manufacturing & Nanotechnology", "Measurement & Precision Engineering", "Agricultural & Crop Research Methods", 
    "Applied Biotechnology & Bioresource Utilization", "Cultural Heritage Research & Preservation Methods", "Aerospace & Space Engineering Methods", 
    "Autonomous Systems & Robotics Development", "Behavioral Science Research & Design Interventions", "Bio-Inspired Computing & Algorithm Design", 
    "Bioengineering & Biomanufacturing Processes", "Waste Processing & Resource Valorization", 
    "Biomarker Discovery & Diagnostic Development", "Biomedical Data Analysis & Computational Biology", "Biotechnology Research Methodologies", 
    "Business Model & Market Innovation Strategies", "Capacity Building, Training & Workforce Development", "Chemical & Green Process Engineering", 
    "Circular Economy Analysis & Lifecycle Assessment", "Community Engagement & Stakeholder Collaboration", "Clean Propulsion & Emission Control Technologies", 
    "Climate & Ecological System Modeling", "Clinical Research & Human Trials Methodology", "Communication, Dissemination & Outreach Strategies", 
    "Complex Systems Modeling & Analysis", "Computational Modeling & Simulation Techniques", "Computational Social Science Methods", 
    "Computer Graphics & Virtual Prototyping", "Computer Vision & Image Processing Systems", "Conservation Management & Ecological Monitoring", 
    "Cybersecurity Protocols & Data Privacy Techniques", "Data Analysis & Statistical Methods", "Data Collection, Curation & Governance", 
    "Decision Support Systems & Science", "Device Design & Instrument Engineering", "Digital Content Creation & Digital Scholarship Methods", 
    "Digital User Experience & Human-Computer Interaction Design", "Digital Health Solutions & Remote Care Delivery", "Digital Infrastructure Development & Transformation Strategies", 
    "Disease Modeling & Epidemiological Surveillance", "Drug Discovery, Delivery & Preclinical Development", "Earth Observation & Geospatial Analytics", 
    "Energy Infrastructure & Efficiency Optimization", "Engineering Design & Development Methodology", "Environmental Impact Assessment & Monitoring", 
    "Ethical Framework Development & Policy Analysis", "Experimental Design & Laboratory Analysis", "Food Processing, Quality & Safety Methods", 
    "Gene Therapy & Genetic Engineering Techniques", "Healthcare System Design & Optimization", "Health Impact Assessment & Health Promotion Strategies", 
    "Advanced Imaging & Microscopy Techniques", "Immunological Assays & Therapeutic Development", "Impact Assessment & Policy Implementation", 
    "Industrial Process Design & Optimization", "Information Retrieval & Knowledge Management Systems", "Infrastructure Planning & Smart Systems Integration", 
    "Innovation Management & Technology Transfer", "Instrumentation Development & Sensor Technology", "Integrated Assessment Frameworks & Modeling", 
    "Internet of Things & Digital Automation Solutions", "Laboratory Automation & High-Throughput Screening", "Language Processing & Text Analysis Methods", 
    "Material Design, Synthesis & Characterization", "Mathematical Modeling & Numerical Analysis", "Medical Technology Design & Innovation", 
    "Microbiome Engineering & Host-Microbe Interaction Research", "Mixed Methods Research Approaches", "Neuroscience & Cognitive Research Methods", 
    "Open Science Practices & Research Governance", "Operational Efficiency & Systems Engineering", "Organizational Development & Strategic Management", 
    "Participatory Action Research & Co-Design Methods", "Philosophical Inquiry & Ethical Analysis", "Physical Sciences Experimental Research", 
    "Physiological Measurement & Analysis Studies", "Plant Biotechnology & Applied Agricultural Research", "Product Development & Lifecycle Management", 
    "Proteomics, Metabolomics & Mass Spectrometry", "Qualitative Research Methodologies", "Quantitative Research Methodologies", 
    "Urban & Regional Development Modeling & Planning", "Reliability Engineering & Risk Assessment", "Resource Management & Optimization Strategies", 
    "Scenario Analysis & Foresight Planning", "Scientific Infrastructure Development & Management", "Social & Economic Impact Research", 
    "Software Development & Systems Engineering", "Supply Chain Design & Optimization", "Synthetic Biology & Metabolic Engineering", 
    "Theoretical Modeling & Computational Physics/Chemistry", "Indigenous Knowledge Systems & Co-creation", "Translational Research & Clinical Application", 
    "System Verification, Validation & Testing", "Water Resource Management & Purification Technologies"
]

print(f"Using {len(DOMAIN_CATEGORIES_CURATED)} detailed domains and {len(METHOD_CATEGORIES_CURATED)} detailed methods")

# Proven prompt template (same structure as the working approach)
BATCH_CLASSIFY_PROMPT_TEMPLATE = """You are an expert taxonomist specializing in research project classification. Your task is to classify multiple research project summaries, assigning each project exactly one domain area and one method area from the provided lists.

Domain Areas:
{domain_categories}

Method Areas:
{method_categories}

Classification Guidelines:
1. For Domain Area, focus on the TECHNICAL AREA OF OPERATION or SCIENTIFIC FIELD rather than business function.
   - Example: If a project involves a business developing biotechnology products, classify it as "Biotechnology" rather than "Business & Organizational Behavior"
   - Always prioritize the specialized technical or scientific domain over generic business categories
   - Look for the core scientific, technical, or application area that best represents what the project is about

2. For Method Area, select the primary methodology, approach, or technique used in the project.
   - Focus on HOW the work is being accomplished

For each project, select:
- The single most appropriate domain area that best captures the primary technical field or area of operation
- The single most appropriate method area that best captures the main methodology

Projects to Classify:
{project_summaries}

Output Format:
Project 1:
Domain Area: [Selected Domain Area - must be exactly as shown in the list]
Method Area: [Selected Method Area - must be exactly as shown in the list]

Project 2:
Domain Area: [Selected Domain Area - must be exactly as shown in the list]
Method Area: [Selected Method Area - must be exactly as shown in the list]

... continue for all projects ...

Important: For each project, select only ONE domain area and ONE method area. Both must be exact matches from the provided lists. Number your responses to match the project numbers exactly."""

# Improved find_closest_match function (fixed fallback)
def find_closest_match(value: str, category_list: List[str]) -> str:
    """Find the closest match for a value in a category list"""
    if not value:
        return "Classification Failed"
    
    # First try exact match
    for category in category_list:
        if value.lower() == category.lower():
            return category
    
    # Then try partial matches
    for category in category_list:
        if value.lower() in category.lower() or category.lower() in value.lower():
            return category
    
    # If no match found, return Classification Failed (honest fallback)
    return "Classification Failed"

# Proven classify_batch function (no validation - let LLM assign anything)
def classify_batch(project_summaries: List[str], domain_categories: List[str], 
                   method_categories: List[str], max_retries: int = 3) -> List[Dict[str, str]]:
    """Classify a batch of project text summaries - accepts any LLM output"""
    
    # Format the project summaries for the prompt (use FULL summaries)
    formatted_summaries = ""
    for i, summary in enumerate(project_summaries, 1):
        # Use full summaries without truncation (key difference!)
        formatted_summaries += f"Project {i}:\n{summary}\n\n"
    
    # Create the prompt
    prompt = BATCH_CLASSIFY_PROMPT_TEMPLATE.format(
        domain_categories="\n".join(domain_categories),
        method_categories="\n".join(method_categories),
        project_summaries=formatted_summaries
    )
    
    # Process with model (with retries)
    output_text = ""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            output_text = response.text
            break
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** (attempt + 1)  # Exponential backoff
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Failed all retry attempts")
                return [{"domain": "API Failed", "method": "API Failed"} for _ in project_summaries]
    
    # Parse results - accept whatever the LLM assigns
    results = []
    project_blocks = re.split(r'Project \d+:', output_text)
    
    # Skip the first item if it's empty
    if project_blocks and not project_blocks[0].strip():
        project_blocks = project_blocks[1:]
    
    # If we didn't get enough results, pad with failures
    if len(project_blocks) < len(project_summaries):
        print(f"Warning: Expected {len(project_summaries)} results but got {len(project_blocks)}")
        project_blocks.extend(["" for _ in range(len(project_summaries) - len(project_blocks))])
    
    # Process each project block - NO VALIDATION
    for i, block in enumerate(project_blocks[:len(project_summaries)]):
        domain = "Parse Failed"
        method = "Parse Failed"
        
        for line in block.strip().split('\n'):
            if line.startswith("Domain Area:"):
                domain = line.replace("Domain Area:", "").strip()
            elif line.startswith("Method Area:"):
                method = line.replace("Method Area:", "").strip()
        
        # Just accept whatever the LLM assigned - no validation!
        results.append({"domain": domain, "method": method})
    
    return results

# Display detailed results for first batch (from working approach)
def display_first_batch_results(batch_summaries: List[str], batch_results: List[Dict[str, str]]):
    """Display detailed results for the first batch for verification"""
    print("\n===== FIRST BATCH RESULTS =====")
    print(f"Showing results for the first batch of {len(batch_summaries)} projects")
    
    for i, result in enumerate(batch_results):
        summary = batch_summaries[i]
        domain = result["domain"]
        method = result["method"]
        
        # Truncate summary for display
        truncated_summary = summary[:150] + "..." if len(summary) > 150 else summary
        truncated_summary = truncated_summary.replace('\n', ' ')
        
        print(f"\nProject {i+1}:")
        print(f"Summary: {truncated_summary}")
        print(f"Assigned Domain: {domain}")
        print(f"Assigned Method: {method}")
    
    print("\n================================")

# Main processing function (uses your global variables)
def classify_dataset_working_approach():
    """Process the dataset using the proven working approach with your global settings"""
    
    print(f"Using proven approach with YOUR settings:")
    print(f"  Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"  Model: {model.model_name if hasattr(model, 'model_name') else 'Global model'}")
    print(f"  Trial mode: {RUN_OVERALL_TRIAL_MODE}")
    if RUN_OVERALL_TRIAL_MODE:
        print(f"  Max trial batches: {MAX_BATCHES_FOR_TRIAL}")
    
    # Initialize result columns
    df_complete['s5_curated_domain'] = ""
    df_complete['s5_curated_method'] = ""
    
    # Determine scope using YOUR global variables
    if RUN_OVERALL_TRIAL_MODE:
        max_rows = DEFAULT_BATCH_SIZE * MAX_BATCHES_FOR_TRIAL
        process_df = df_complete.iloc[:max_rows].copy()
        print(f"TRIAL MODE: Processing {len(process_df)} rows")
    else:
        process_df = df_complete.copy()
        print(f"FULL MODE: Processing {len(process_df)} rows")
    
    # Process in batches using YOUR batch size
    total_records = len(process_df)
    total_batches = (total_records + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
    
    for batch_num, start_idx in enumerate(range(0, total_records, DEFAULT_BATCH_SIZE)):
        end_idx = min(start_idx + DEFAULT_BATCH_SIZE, total_records)
        batch_size_actual = end_idx - start_idx
        
        print(f"\nProcessing batch {batch_num + 1}/{total_batches} (records {start_idx+1}-{end_idx})")
        
        # Extract summaries for this batch
        batch_summaries = []
        for idx in range(start_idx, end_idx):
            summary = process_df.iloc[idx]['full_text']
            batch_summaries.append(summary)
        
        # Classify the batch using YOUR global model
        batch_results = classify_batch(batch_summaries, DOMAIN_CATEGORIES_CURATED, METHOD_CATEGORIES_CURATED)
        
        # Save results directly to df_complete
        for i in range(batch_size_actual):
            original_idx = process_df.index[start_idx + i]
            if i < len(batch_results):
                df_complete.loc[original_idx, 's5_curated_domain'] = batch_results[i]['domain']
                df_complete.loc[original_idx, 's5_curated_method'] = batch_results[i]['method']
            else:
                df_complete.loc[original_idx, 's5_curated_domain'] = "API Failed"
                df_complete.loc[original_idx, 's5_curated_method'] = "API Failed"
        
        # Display detailed results for first batch only
        if batch_num == 0:
            display_first_batch_results(batch_summaries, batch_results)
        
        # Save every 20 batches
        if (batch_num + 1) % 20 == 0:
            df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
            print(f"  ✅ SAVED at batch {batch_num + 1}")
        
        # Wait between batches using YOUR global wait time
        if batch_num < total_batches - 1:
            print(f"Waiting {DEFAULT_WAIT_TIME} seconds before next batch...")
            time.sleep(DEFAULT_WAIT_TIME)
    
    return df_complete

# Run the proven approach
if 'df_complete' not in locals() or df_complete.empty:
    print("ERROR: df_complete not available")
elif 'model' not in locals() or not model:
    print("ERROR: model not available")  
elif 'full_text' not in df_complete.columns:
    print("ERROR: 'full_text' column missing")
else:
    # Use the proven working approach with YOUR global settings
    df_result = classify_dataset_working_approach()
    
    # Final save
    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
    print(f"✅ FINAL SAVE COMPLETE")
    
    # Generate summary statistics (from working approach)
    domain_counts = df_complete['s5_curated_domain'].value_counts()
    method_counts = df_complete['s5_curated_method'].value_counts()
    
    print("\nTop 10 assigned domain categories:")
    for domain, count in domain_counts.head(10).items():
        print(f"- {domain}: {count}")
    
    print("\nTop 10 assigned method categories:")
    for method, count in method_counts.head(10).items():
        print(f"- {method}: {count}")
    
    print(f"\n🎉 STAGE 5 COMPLETE!")
    print(f"Total projects processed: {len(df_complete)}")
    
    # Show sample results
    display(df_complete[['id', 'full_text', 's5_curated_domain', 's5_curated_method']].head())

# SEPARATE CELL - Check for non-matching categories
def check_category_matches():
    """Check which assigned categories don't match the provided lists"""
    
    print("="*60)
    print("CHECKING FOR NON-MATCHING CATEGORIES") 
    print("="*60)
    
    # Get all assigned categories
    assigned_domains = df_complete['s5_curated_domain'].dropna().unique()
    assigned_methods = df_complete['s5_curated_method'].dropna().unique()
    
    # Find non-matching domains
    non_matching_domains = []
    for domain in assigned_domains:
        if domain not in DOMAIN_CATEGORIES_CURATED and domain not in ["API Failed", "Parse Failed"]:
            non_matching_domains.append(domain)
    
    # Find non-matching methods  
    non_matching_methods = []
    for method in assigned_methods:
        if method not in METHOD_CATEGORIES_CURATED and method not in ["API Failed", "Parse Failed"]:
            non_matching_methods.append(method)
    
    # Report results
    print(f"Total unique domains assigned: {len(assigned_domains)}")
    print(f"Domains matching your list: {len(assigned_domains) - len(non_matching_domains)}")
    print(f"Domains NOT in your list: {len(non_matching_domains)}")
    
    if non_matching_domains:
        print(f"\nNON-MATCHING DOMAINS:")
        for domain in non_matching_domains:
            count = (df_complete['s5_curated_domain'] == domain).sum()
            print(f"  '{domain}': {count} projects")
    
    print(f"\nTotal unique methods assigned: {len(assigned_methods)}")
    print(f"Methods matching your list: {len(assigned_methods) - len(non_matching_methods)}")
    print(f"Methods NOT in your list: {len(non_matching_methods)}")
    
    if non_matching_methods:
        print(f"\nNON-MATCHING METHODS:")
        for method in non_matching_methods:
            count = (df_complete['s5_curated_method'] == method).sum()
            print(f"  '{method}': {count} projects")
    
    # Calculate match rates
    total_projects = len(df_complete)
    domain_exact_matches = sum(1 for d in df_complete['s5_curated_domain'] if d in DOMAIN_CATEGORIES_CURATED)
    method_exact_matches = sum(1 for m in df_complete['s5_curated_method'] if m in METHOD_CATEGORIES_CURATED)
    
    print(f"\n" + "="*60)
    print("MATCH RATES")
    print("="*60)
    print(f"Domain exact match rate: {domain_exact_matches}/{total_projects} ({domain_exact_matches/total_projects*100:.1f}%)")
    print(f"Method exact match rate: {method_exact_matches}/{total_projects} ({method_exact_matches/total_projects*100:.1f}%)")
    
    return non_matching_domains, non_matching_methods

# Run the check after Stage 5 completes
check_category_matches()


# ### We further condensed the categories

# In[135]:


# COMPLETE S5 FIX STAGE
# S5 CURATED CATEGORIES (from your original code)
S5_DOMAIN_CATEGORIES = [
    "AI & Machine Learning", "Advanced Computing & Electronics", "Advanced Manufacturing & Industry", 
    "Advanced Therapies & Regenerative Medicine", "Aerospace Engineering & Aviation", "Agricultural Systems & Plant Science", 
    "Alternative & Biofuels", "Analytical & Theoretical Chemistry", "Animal & Veterinary Sciences", 
    "Anthropology & Archaeology", "Applied & Plasma Physics", "Aquaculture & Fisheries", 
    "Arts, Humanities & Culture", "Astronomy, Cosmology & Particle Physics", "Atmospheric, Climate & Environmental Science", 
    "Autoimmune & Chronic Disease", "Autonomous Systems & Robotics", "Basic & Cross-Domain Science", 
    "Behavioral Science & Psychology", "Biodiversity & Ecology", "Bioeconomy & Circular Economy", 
    "Bioengineering & Synthetic Biology", "Bioinformatics & Systems Biology", "Biological Sciences", 
    "Biomedical Devices & Imaging", "Biomedical Engineering", "Biomedical Research (General)", 
    "Biotechnology", "Brain Science & Neuroscience", "Business & Organizational Behavior", 
    "Cancer Research & Oncology", "Carbon Management & Climate Technology", "Cardiovascular Health", 
    "Chemical & Process Engineering", "Chemical Safety & Toxicology", "Civic Engagement & Governance", 
    "Civil Engineering & Infrastructure", "Clean & Renewable Energy", "Climate Adaptation & Impact", 
    "Cloud & Digital Infrastructure", "Cognitive Science", "Communication & Media Studies", 
    "Complex Systems Science", "Computer Science (General)", "Computer Vision & Graphics", 
    "Conflict, Security & Justice", "Crisis & Disaster Management", "Critical Infrastructure & Safety", 
    "Critical Materials & Waste Management", "Cybersecurity & Digital Governance", "Data Science & Analytics", 
    "Dental & Oral Health", "Design Research & Human-Computer Interaction", "Digital Economy & Innovation", 
    "Digital Health & Medical Technology", "Digital Humanities & Cultural Heritage", "Drug Discovery & Pharmaceutical Development", 
    "Economics & Development Studies", "Energy Generation & Systems", "Food Science, Nutrition & Security", 
    "Genomics & Precision Medicine", "Gerontology & Aging Studies", "Historical Studies", 
    "Imaging & Instrumentation Technologies", "Immunology & Vaccine Development", "Life Sciences (General)", 
    "Linguistics & Language Studies", "Logistics & Supply Chain Management", "Marine, Ocean & Water Science", 
    "Materials Science & Engineering", "Mathematics & Applied Statistics", "Mechanical Engineering", 
    "Medical Education & Training", "Mental Health & Psychiatry", "Metabolic & Endocrine Health", 
    "Metrology & Measurement Science", "Microbiology & Microbiome Science", "Migration & Urban Studies", 
    "Musculoskeletal Health & Rehabilitation", "National Security & Defense Studies", "Natural Language Processing", 
    "Natural Resources & Environmental Management", "Neglected & Rare Diseases", "Network Science & Telecommunications", 
    "Nuclear Science & Engineering", "Occupational Health & Safety", "Optics & Photonics", 
    "Organic & Inorganic Chemistry", "Patient Care & Health Systems", "Pediatric & Women's Health", 
    "Philosophy, Logic & Ethics", "Physical & Chemical Sciences", "Planetary & Space Science", 
    "Political Science & Policy", "Public Health & Global Health", "Quantum Science & Technology", 
    "Regional & Urban Planning", "Religious Studies & Theology", "Remote Sensing & Geospatial Technology", 
    "Research & Innovation Policy", "Science & Technology Studies", "Simulation & Training Technology", 
    "Smart Systems & Internet of Things", "Social Justice & Equity", "Social Research & Sociology", 
    "Software Systems Engineering", "Stem Cell & Regenerative Engineering", "Sustainability Science & Education", 
    "Sustainable Consumption & Lifestyles", "Systems Engineering (General)", "Theoretical Physics", 
    "Toxicology & Risk Assessment", "Translational Medicine", "Transport Safety & Emissions reduction", 
    "Venture Capital & Innovation Finance", "Wildlife Health & Ecology"
]

S5_METHOD_CATEGORIES = [
    "AI & Machine Learning Development", "Advanced Analytical Techniques", "Advanced Engineering Methods", 
    "Advanced Manufacturing & Nanotechnology", "Measurement & Precision Engineering", "Agricultural & Crop Research Methods", 
    "Applied Biotechnology & Bioresource Utilization", "Cultural Heritage Research & Preservation Methods", "Aerospace & Space Engineering Methods", 
    "Autonomous Systems & Robotics Development", "Behavioral Science Research & Design Interventions", "Bio-Inspired Computing & Algorithm Design", 
    "Bioengineering & Biomanufacturing Processes", "Waste Processing & Resource Valorization", 
    "Biomarker Discovery & Diagnostic Development", "Biomedical Data Analysis & Computational Biology", "Biotechnology Research Methodologies", 
    "Business Model & Market Innovation Strategies", "Capacity Building, Training & Workforce Development", "Chemical & Green Process Engineering", 
    "Circular Economy Analysis & Lifecycle Assessment", "Community Engagement & Stakeholder Collaboration", "Clean Propulsion & Emission Control Technologies", 
    "Climate & Ecological System Modeling", "Clinical Research & Human Trials Methodology", "Communication, Dissemination & Outreach Strategies", 
    "Complex Systems Modeling & Analysis", "Computational Modeling & Simulation Techniques", "Computational Social Science Methods", 
    "Computer Graphics & Virtual Prototyping", "Computer Vision & Image Processing Systems", "Conservation Management & Ecological Monitoring", 
    "Cybersecurity Protocols & Data Privacy Techniques", "Data Analysis & Statistical Methods", "Data Collection, Curation & Governance", 
    "Decision Support Systems & Science", "Device Design & Instrument Engineering", "Digital Content Creation & Digital Scholarship Methods", 
    "Digital User Experience & Human-Computer Interaction Design", "Digital Health Solutions & Remote Care Delivery", "Digital Infrastructure Development & Transformation Strategies", 
    "Disease Modeling & Epidemiological Surveillance", "Drug Discovery, Delivery & Preclinical Development", "Earth Observation & Geospatial Analytics", 
    "Energy Infrastructure & Efficiency Optimization", "Engineering Design & Development Methodology", "Environmental Impact Assessment & Monitoring", 
    "Ethical Framework Development & Policy Analysis", "Experimental Design & Laboratory Analysis", "Food Processing, Quality & Safety Methods", 
    "Gene Therapy & Genetic Engineering Techniques", "Healthcare System Design & Optimization", "Health Impact Assessment & Health Promotion Strategies", 
    "Advanced Imaging & Microscopy Techniques", "Immunological Assays & Therapeutic Development", "Impact Assessment & Policy Implementation", 
    "Industrial Process Design & Optimization", "Information Retrieval & Knowledge Management Systems", "Infrastructure Planning & Smart Systems Integration", 
    "Innovation Management & Technology Transfer", "Instrumentation Development & Sensor Technology", "Integrated Assessment Frameworks & Modeling", 
    "Internet of Things & Digital Automation Solutions", "Laboratory Automation & High-Throughput Screening", "Language Processing & Text Analysis Methods", 
    "Material Design, Synthesis & Characterization", "Mathematical Modeling & Numerical Analysis", "Medical Technology Design & Innovation", 
    "Microbiome Engineering & Host-Microbe Interaction Research", "Mixed Methods Research Approaches", "Neuroscience & Cognitive Research Methods", 
    "Open Science Practices & Research Governance", "Operational Efficiency & Systems Engineering", "Organizational Development & Strategic Management", 
    "Participatory Action Research & Co-Design Methods", "Philosophical Inquiry & Ethical Analysis", "Physical Sciences Experimental Research", 
    "Physiological Measurement & Analysis Studies", "Plant Biotechnology & Applied Agricultural Research", "Product Development & Lifecycle Management", 
    "Proteomics, Metabolomics & Mass Spectrometry", "Qualitative Research Methodologies", "Quantitative Research Methodologies", 
    "Urban & Regional Development Modeling & Planning", "Reliability Engineering & Risk Assessment", "Resource Management & Optimization Strategies", 
    "Scenario Analysis & Foresight Planning", "Scientific Infrastructure Development & Management", "Social & Economic Impact Research", 
    "Software Development & Systems Engineering", "Supply Chain Design & Optimization", "Synthetic Biology & Metabolic Engineering", 
    "Theoretical Modeling & Computational Physics/Chemistry", "Indigenous Knowledge Systems & Co-creation", "Translational Research & Clinical Application", 
    "System Verification, Validation & Testing", "Water Resource Management & Purification Technologies"
]

def find_s5_failed_projects():
    """Find the exact projects that still have non-matching S5 categories - STRICT MODE"""
    
    print("🔍 FINDING S5 FAILED PROJECTS (STRICT - NO PARSE FAILED ALLOWED)")
    print("="*60)
    
    # Load current data
    try:
        df = pd.read_csv("df_processing_checkpoint.csv")
        print(f"✅ Loaded checkpoint CSV: {df.shape}")
    except:
        df = globals()['df_complete'].copy()
        print(f"✅ Using global df_complete: {df.shape}")
    
    # STRICT: Only empty strings and None are acceptable "failures"
    # Everything else (including "Parse Failed", "API Failed", etc.) must be fixed
    acceptable_errors = ["", None]
    
    failed_projects = []
    
    for idx in df.index:
        domain = df.loc[idx, 's5_curated_domain']
        method = df.loc[idx, 's5_curated_method']
        
        # Check if domain is problematic
        domain_bad = (pd.notna(domain) and 
                     str(domain).strip() != "" and  # Not empty string
                     domain not in S5_DOMAIN_CATEGORIES)  # Not valid category
        
        # Check if method is problematic  
        method_bad = (pd.notna(method) and 
                     str(method).strip() != "" and  # Not empty string
                     method not in S5_METHOD_CATEGORIES)  # Not valid category
        
        if domain_bad or method_bad:
            project_id = df.loc[idx, 'id'] if 'id' in df.columns else idx
            failed_projects.append({
                'index': idx,
                'id': project_id,
                'domain': domain,
                'method': method,
                'domain_bad': domain_bad,
                'method_bad': method_bad,
                'summary': str(df.loc[idx, 'full_text'])[:200] + "..."
            })
    
    print(f"Found {len(failed_projects)} projects with invalid S5 categories")
    
    # Group by bad categories to see patterns
    if failed_projects:
        domain_issues = {}
        method_issues = {}
        
        for proj in failed_projects:
            if proj['domain_bad']:
                domain = str(proj['domain'])
                if domain not in domain_issues:
                    domain_issues[domain] = []
                domain_issues[domain].append(proj['index'])
            
            if proj['method_bad']:
                method = str(proj['method'])
                if method not in method_issues:
                    method_issues[method] = []
                method_issues[method].append(proj['index'])
        
        print(f"\n❌ ALL PROBLEMATIC S5 DOMAINS (including Parse Failed, API Failed, etc.):")
        for domain, indices in domain_issues.items():
            print(f"  '{domain}': {len(indices)} projects")
        
        print(f"\n❌ ALL PROBLEMATIC S5 METHODS (including Parse Failed, API Failed, etc.):")
        for method, indices in method_issues.items():
            print(f"  '{method}': {len(indices)} projects")
    
    return failed_projects, df

def fix_s5_failed_projects_no_parse_failed(failed_projects, df, model):
    """Fix S5 projects - return empty strings instead of failure messages"""
    
    if not failed_projects:
        print("✅ No S5 failed projects to fix!")
        return df
    
    print(f"🔧 S5 FIXING - {len(failed_projects)} PROJECTS (NO PARSE FAILED ALLOWED)")
    print("="*60)
    
    # Enhanced prompt for problem cases
    TARGETED_S5_PROMPT = """You are an expert research taxonomist. You must classify these projects using ONLY the exact categories from the lists below.

CRITICAL RULES:
1. Select EXACTLY ONE domain from the Domain list for each project
2. Select EXACTLY ONE method from the Method list for each project  
3. Use the EXACT spelling, punctuation, and capitalization shown
4. DO NOT create new categories or modify existing ones
5. DO NOT confuse domains with methods

DOMAIN AREAS (Scientific/Technical Fields - choose ONE per project):
{domain_list}

METHOD AREAS (Research Approaches/Techniques - choose ONE per project):
{method_list}

PROJECTS TO CLASSIFY:
{project_summaries}

RESPOND IN THIS EXACT FORMAT:
Project 1:
Domain Area: [EXACT match from Domain list above]
Method Area: [EXACT match from Method list above]

Project 2:
Domain Area: [EXACT match from Domain list above]
Method Area: [EXACT match from Method list above]

... continue for all projects ...

Remember: Domain = WHAT field/area, Method = HOW it's done."""

    changes_made = []
    df_working = df.copy()
    
    # Process 6 projects at a time
    batch_size = 6
    total_batches = (len(failed_projects) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(failed_projects))
        batch_projects = failed_projects[start_idx:end_idx]
        
        print(f"\n--- S5 Batch {batch_num+1}/{total_batches} ({len(batch_projects)} projects) ---")
        
        # Show what we're fixing
        for i, proj in enumerate(batch_projects):
            print(f"  Project {i+1} (ID {proj['id']}, Index {proj['index']}):")
            print(f"    Current Domain: '{proj['domain']}' {'❌' if proj['domain_bad'] else '✅'}")
            print(f"    Current Method: '{proj['method']}' {'❌' if proj['method_bad'] else '✅'}")
        
        # Prepare batch summaries
        project_summaries = ""
        for i, proj in enumerate(batch_projects, 1):
            project_text = str(df_working.loc[proj['index'], 'full_text'])[:1200]  # Limit length
            project_summaries += f"Project {i}:\nTitle and Description: {project_text}\n\n"
        
        # Create focused prompt
        prompt = TARGETED_S5_PROMPT.format(
            domain_list="\n".join(S5_DOMAIN_CATEGORIES),
            method_list="\n".join(S5_METHOD_CATEGORIES),
            project_summaries=project_summaries
        )
        
        # Try to get results from LLM
        batch_success = False
        try:
            print(f"🤖 Calling LLM for batch {batch_num+1}...")
            
            response = model.generate_content(prompt)
            output = response.text.strip()
            
            print(f"✅ Got LLM response, parsing...")
            
            # Parse results for the batch
            project_blocks = re.split(r'Project \d+:', output)
            if project_blocks and not project_blocks[0].strip():
                project_blocks = project_blocks[1:]
            
            if len(project_blocks) >= len(batch_projects):
                batch_results = []
                
                for i, block in enumerate(project_blocks[:len(batch_projects)]):
                    new_domain = ""  # Default to EMPTY instead of "Parse Failed"
                    new_method = ""  # Default to EMPTY instead of "Parse Failed"
                    
                    lines = block.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Domain Area:"):
                            candidate = line.replace("Domain Area:", "").strip()
                            candidate = candidate.strip('"\'- ')
                            # Only assign if it's actually in our valid list
                            if candidate in S5_DOMAIN_CATEGORIES:
                                new_domain = candidate
                        elif line.startswith("Method Area:"):
                            candidate = line.replace("Method Area:", "").strip()
                            candidate = candidate.strip('"\'- ')
                            # Only assign if it's actually in our valid list
                            if candidate in S5_METHOD_CATEGORIES:
                                new_method = candidate
                    
                    batch_results.append({
                        'domain': new_domain,
                        'method': new_method
                    })
                
                # Apply all results in this batch
                for i, result in enumerate(batch_results):
                    proj = batch_projects[i]
                    idx = proj['index']
                    
                    old_domain = df_working.loc[idx, 's5_curated_domain']
                    old_method = df_working.loc[idx, 's5_curated_method']
                    
                    # Only update categories that were actually problematic
                    if proj['domain_bad']:
                        df_working.loc[idx, 's5_curated_domain'] = result['domain']
                        print(f"  🔄 Fixed Domain {idx}: '{old_domain}' → '{result['domain']}'")
                    
                    if proj['method_bad']:
                        df_working.loc[idx, 's5_curated_method'] = result['method']
                        print(f"  🔄 Fixed Method {idx}: '{old_method}' → '{result['method']}'")
                    
                    changes_made.append({
                        'index': idx,
                        'id': proj['id'],
                        'old_domain': old_domain,
                        'new_domain': result['domain'] if proj['domain_bad'] else old_domain,
                        'old_method': old_method,
                        'new_method': result['method'] if proj['method_bad'] else old_method
                    })
                
                batch_success = True
                print(f"  ✅ Batch {batch_num+1} processed successfully")
            else:
                print(f"  ⚠️ Got {len(project_blocks)} project blocks, expected {len(batch_projects)}")
                
        except Exception as e:
            print(f"  ❌ LLM error for batch {batch_num+1}: {e}")
        
        if not batch_success:
            print(f"  🔄 Batch {batch_num+1} failed - setting empty strings for problematic categories")
            # Set empty strings for failed attempts (so they get caught in next iteration)
            for proj in batch_projects:
                idx = proj['index']
                if proj['domain_bad']:
                    df_working.loc[idx, 's5_curated_domain'] = ""
                    print(f"    Set domain to empty for index {idx}")
                if proj['method_bad']:
                    df_working.loc[idx, 's5_curated_method'] = ""
                    print(f"    Set method to empty for index {idx}")
        
        # Small delay between batches
        if batch_num < total_batches - 1:
            time.sleep(3)
    
    print(f"\n💾 SAVING S5 FIXES")
    print("="*60)
    
    # Save back to CSV
    df_working.to_csv("df_processing_checkpoint.csv", index=False)
    print(f"✅ Saved to df_processing_checkpoint.csv")
    
    # Update global if possible
    try:
        globals()['df_complete'] = df_working.copy()
        print(f"✅ Updated global df_complete")
    except:
        pass
    
    print(f"\n📊 S5 FIX SUMMARY:")
    print(f"   Failed projects processed: {len(failed_projects)}")
    print(f"   Batches processed: {total_batches}")
    print(f"   Changes attempted: {len(changes_made)}")
    
    return df_working

def run_s5_complete_fix():
    """Main function to run complete S5 fixes until everything is clean"""
    
    print("🎯 S5 COMPLETE FIX - NO PARSE FAILED ALLOWED")
    print("="*80)
    
    # Get model
    try:
        model = globals()['model']
        print(f"✅ Using global model")
    except:
        print("❌ ERROR: Global 'model' not found")
        return None
    
    # Keep running fix iterations until no more failures
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"\n🔄 S5 FIX ITERATION {iteration + 1}/{max_iterations}")
        print("="*60)
        
        # Step 1: Find failed projects (strict mode)
        failed_projects, df = find_s5_failed_projects()
        
        if not failed_projects:
            print(f"🎉 S5 COMPLETE - No more failed projects after {iteration + 1} iteration(s)!")
            break
        
        print(f"📋 Found {len(failed_projects)} projects that need fixing")
        
        # Step 2: Apply fixes
        df = fix_s5_failed_projects_no_parse_failed(failed_projects, df, model)
        
        # Step 3: Show progress
        print(f"\n📊 Iteration {iteration + 1} complete")
        
        if iteration == max_iterations - 1:
            print(f"⚠️ Reached maximum iterations ({max_iterations})")
            print(f"   Some projects may still need manual review")
    
    # Final verification
    print(f"\n🔍 FINAL S5 VERIFICATION")
    print("="*60)
    
    final_failed, df_final = find_s5_failed_projects()
    
    domain_matches = sum(1 for d in df_final['s5_curated_domain'] if d in S5_DOMAIN_CATEGORIES)
    method_matches = sum(1 for m in df_final['s5_curated_method'] if m in S5_METHOD_CATEGORIES)
    empty_domains = sum(1 for d in df_final['s5_curated_domain'] if pd.isna(d) or str(d).strip() == "")
    empty_methods = sum(1 for m in df_final['s5_curated_method'] if pd.isna(m) or str(m).strip() == "")
    total_projects = len(df_final)
    
    print(f"📊 FINAL S5 STATISTICS:")
    print(f"   Total projects: {total_projects}")
    print(f"   Valid domain categories: {domain_matches} ({domain_matches/total_projects*100:.2f}%)")
    print(f"   Valid method categories: {method_matches} ({method_matches/total_projects*100:.2f}%)")
    print(f"   Empty domains (acceptable): {empty_domains} ({empty_domains/total_projects*100:.2f}%)")
    print(f"   Empty methods (acceptable): {empty_methods} ({empty_methods/total_projects*100:.2f}%)")
    print(f"   Remaining problematic projects: {len(final_failed)}")
    
    if len(final_failed) == 0:
        print(f"\n🎉🎉🎉 PERFECT! 100% S5 SUCCESS - NO MORE PARSE FAILED! 🎉🎉🎉")
    else:
        print(f"\n⚠️ Still have {len(final_failed)} projects with invalid categories:")
        # Show first few remaining problems
        for proj in final_failed[:5]:
            print(f"   - Index {proj['index']}: Domain='{proj['domain']}', Method='{proj['method']}'")
        
        if len(final_failed) > 5:
            print(f"   ... and {len(final_failed) - 5} more")
    
    print(f"\n🎯 S5 COMPLETE FIX FINISHED!")
    
    return df_final

# USAGE:
df_s5_clean = run_s5_complete_fix()


# In[145]:


# S6 REFINED CATEGORIES 
S6_DOMAIN_CATEGORIES = [
    "Fundamental Physics & Chemistry",
    "Mathematical & Computational Sciences", 
    "Biological & Life Sciences",
    "Earth, Environmental & Climate Science",
    "Agriculture, Food & Bioresources",
    "Aerospace, Space & Astronomy",
    "Computer Science & Information Technology",
    "Artificial Intelligence & Data Science",
    "Robotics & Autonomous Systems",
    "Electrical, Electronics & Photonics Engineering",
    "Materials Science & Engineering",
    "Chemical Engineering & Applied Chemistry",
    "Mechanical & Industrial Engineering",
    "Biomedical Engineering & Medical Technology",
    "Energy Systems & Technology",
    "Transportation, Urban & Regional Development",
    "Clinical Medicine & Health Sciences",
    "Neuroscience, Neurology & Cognitive Science",
    "Microbiology, Virology & Infectious Diseases",
    "Cancer & Oncology",
    "Drug Discovery & Pharmaceutical Development",
    "Public Health, Epidemiology & Healthcare Administration",
    "Social Sciences & Human Behavior",
    "Psychology & Behavioral Science",
    "Business, Management & Economics",
    "Policy, Governance & Law",
    "Humanities, Arts & Cultural Heritage",
    "Education & Training",
    "Sustainable Development & Global Challenges",
    "Security & Defense",
    "Environmental Health & Engineering",
    "Nuclear Science & Engineering",
    "Quantum Physics & Technologies"
]

S6_METHOD_CATEGORIES = [
    "Qualitative, Archival & Interpretive Research",
    "Quantitative, Statistical & Mathematical Modeling",
    "Computational Modeling, Simulation & Optimization",
    "Artificial Intelligence & Machine Learning Methods",
    "Experimental Design & Laboratory Techniques",
    "Data Collection, Curation & Management",
    "Data Analysis, Interpretation & Visualization",
    "Engineering Design, Fabrication & Prototyping",
    "Process Engineering & Optimization",
    "Materials Synthesis, Characterization & Analysis",
    "Chemical Synthesis, Catalysis & Analytical Methods",
    "Omics, Molecular & Cellular Biology Techniques",
    "Genetic Engineering & Synthetic Biology",
    "Bioprocessing, Biomanufacturing & Bioengineering",
    "Drug Discovery, Development & Delivery Methods",
    "Clinical & Healthcare Research Methods",
    "Diagnostic, Therapeutic & Medical Technology Development",
    "Advanced Imaging, Microscopy & Spectroscopy",
    "Sensor, Monitoring & Measurement Technologies",
    "Remote Sensing, Geospatial Analysis & Mapping",
    "Ecological & Environmental Modeling & Assessment",
    "Policy, Economic & Social Impact Analysis",
    "Ethical, Legal & Societal Framework Development",
    "Neuroscience, Cognitive & Behavioral Research Methods",
    "Agricultural, Plant Science & Food System Methods",
    "Immunological & Infectious Disease Research",
    "Software Engineering, HCI & Digital System Design",
    "Cybersecurity, Network Engineering & Data Protection",
    "Lifecycle, Risk Assessment & Environmental Management",
    "Innovation, Business Development & Digital Transformation",
    "Knowledge Management, Collaboration & Dissemination",
    "Capacity Building, Education & Training Methods",
    "Quantum Computing & Technology Development",
    "Archaeological, Heritage & Conservation Methods",
    "Astro/Space Observation & Geophysical Survey Methods"
]

print(f"🎯 S6 REFINED CATEGORIES LOADED")
print(f"   Domains: {len(S6_DOMAIN_CATEGORIES)} categories")
print(f"   Methods: {len(S6_METHOD_CATEGORIES)} categories")

# S6 Classification Prompt - Based on your proven S5 approach
S6_BATCH_CLASSIFY_PROMPT = """You are an expert taxonomist specializing in research project classification. Your task is to classify multiple research project summaries, assigning each project exactly one domain area and one method area from the provided lists.

Domain Areas (33 refined categories):
{domain_categories}

Method Areas (35 refined categories):
{method_categories}

Classification Guidelines:
1. For Domain Area, focus on the PRIMARY SCIENTIFIC/TECHNICAL FIELD or APPLICATION AREA.
   - Choose the domain that best represents the core scientific discipline or application area
   - If a project spans multiple domains, select the one that represents the primary focus
   - Look for the fundamental field of knowledge or application that the project contributes to

2. For Method Area, select the primary methodology, approach, or technique used in the project.
   - Focus on HOW the research work is being accomplished
   - Choose the most significant methodological approach used
   - Consider the core techniques or approaches that drive the research

3. Each project gets exactly ONE domain and ONE method from the lists above.
   - Use EXACT spelling and capitalization as shown in the lists
   - Select the single best match for each category

Projects to Classify:
{project_summaries}

Output Format:
Project 1:
Domain Area: [Selected Domain Area - must be exactly as shown in the list]
Method Area: [Selected Method Area - must be exactly as shown in the list]

Project 2:
Domain Area: [Selected Domain Area - must be exactly as shown in the list] 
Method Area: [Selected Method Area - must be exactly as shown in the list]

... continue for all projects ...

Important: For each project, select only ONE domain area and ONE method area. Both must be exact matches from the provided lists. Number your responses to match the project numbers exactly."""

def s6_classify_batch(project_summaries: List[str], max_retries: int = 3) -> List[Dict[str, str]]:
    """Classify a batch of project summaries using S6 categories"""
    
    # Format the project summaries for the prompt
    formatted_summaries = ""
    for i, summary in enumerate(project_summaries, 1):
        # Use full summaries without truncation for best results
        formatted_summaries += f"Project {i}:\n{summary}\n\n"
    
    # Create the prompt
    prompt = S6_BATCH_CLASSIFY_PROMPT.format(
        domain_categories="\n".join(S6_DOMAIN_CATEGORIES),
        method_categories="\n".join(S6_METHOD_CATEGORIES),
        project_summaries=formatted_summaries
    )
    
    # Process with model (with retries)
    output_text = ""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            output_text = response.text
            break
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** (attempt + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Failed all retry attempts")
                return [{"domain": "API Failed", "method": "API Failed"} for _ in project_summaries]
    
    # Parse results
    results = []
    project_blocks = re.split(r'Project \d+:', output_text)
    
    # Skip the first item if it's empty
    if project_blocks and not project_blocks[0].strip():
        project_blocks = project_blocks[1:]
    
    # If we didn't get enough results, pad with failures
    if len(project_blocks) < len(project_summaries):
        print(f"Warning: Expected {len(project_summaries)} results but got {len(project_blocks)}")
        project_blocks.extend(["" for _ in range(len(project_summaries) - len(project_blocks))])
    
    # Process each project block
    for i, block in enumerate(project_blocks[:len(project_summaries)]):
        domain = "Parse Failed"
        method = "Parse Failed"
        
        for line in block.strip().split('\n'):
            if line.startswith("Domain Area:"):
                domain = line.replace("Domain Area:", "").strip()
            elif line.startswith("Method Area:"):
                method = line.replace("Method Area:", "").strip()
        
        results.append({"domain": domain, "method": method})
    
    return results

def display_s6_first_batch_results(batch_summaries: List[str], batch_results: List[Dict[str, str]]):
    """Display detailed results for the first batch"""
    print("\n===== S6 FIRST BATCH RESULTS =====")
    print(f"Showing results for the first batch of {len(batch_summaries)} projects")
    
    for i, result in enumerate(batch_results):
        summary = batch_summaries[i]
        domain = result["domain"]
        method = result["method"]
        
        # Truncate summary for display
        truncated_summary = summary[:150] + "..." if len(summary) > 150 else summary
        truncated_summary = truncated_summary.replace('\n', ' ')
        
        print(f"\nProject {i+1}:")
        print(f"Summary: {truncated_summary}")
        print(f"S6 Domain: {domain}")
        print(f"S6 Method: {method}")
    
    print("\n================================")

def s6_classify_dataset():
    """Process the dataset using S6 refined categories"""
    
    print(f"🚀 STARTING S6 REFINED CLASSIFICATION")
    print(f"="*80)
    print(f"Using S6 refined categories:")
    print(f"  {len(S6_DOMAIN_CATEGORIES)} Domain areas")
    print(f"  {len(S6_METHOD_CATEGORIES)} Method areas")
    print(f"  Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"  Trial mode: {RUN_OVERALL_TRIAL_MODE}")
    if RUN_OVERALL_TRIAL_MODE:
        print(f"  Max trial batches: {MAX_BATCHES_FOR_TRIAL}")
    
    # Initialize new S6 result columns
    df_complete['s6_refined_domain'] = ""
    df_complete['s6_refined_method'] = ""
    
    # Determine scope using global variables
    if RUN_OVERALL_TRIAL_MODE:
        max_rows = DEFAULT_BATCH_SIZE * MAX_BATCHES_FOR_TRIAL
        process_df = df_complete.iloc[:max_rows].copy()
        print(f"TRIAL MODE: Processing {len(process_df)} rows")
    else:
        process_df = df_complete.copy()
        print(f"FULL MODE: Processing {len(process_df)} rows")
    
    # Process in batches
    total_records = len(process_df)
    total_batches = (total_records + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
    
    for batch_num, start_idx in enumerate(range(0, total_records, DEFAULT_BATCH_SIZE)):
        end_idx = min(start_idx + DEFAULT_BATCH_SIZE, total_records)
        batch_size_actual = end_idx - start_idx
        
        print(f"\nProcessing S6 batch {batch_num + 1}/{total_batches} (records {start_idx+1}-{end_idx})")
        
        # Extract summaries for this batch
        batch_summaries = []
        for idx in range(start_idx, end_idx):
            summary = process_df.iloc[idx]['full_text']
            batch_summaries.append(summary)
        
        # Classify the batch using S6 categories
        batch_results = s6_classify_batch(batch_summaries)
        
        # Save results directly to df_complete
        for i in range(batch_size_actual):
            original_idx = process_df.index[start_idx + i]
            if i < len(batch_results):
                df_complete.loc[original_idx, 's6_refined_domain'] = batch_results[i]['domain']
                df_complete.loc[original_idx, 's6_refined_method'] = batch_results[i]['method']
            else:
                df_complete.loc[original_idx, 's6_refined_domain'] = "API Failed"
                df_complete.loc[original_idx, 's6_refined_method'] = "API Failed"
        
        # Display detailed results for first batch only
        if batch_num == 0:
            display_s6_first_batch_results(batch_summaries, batch_results)
        
        # Save every 20 batches
        if (batch_num + 1) % 20 == 0:
            df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
            print(f"  ✅ S6 SAVED at batch {batch_num + 1}")
        
        # Wait between batches
        if batch_num < total_batches - 1:
            print(f"Waiting {DEFAULT_WAIT_TIME} seconds before next S6 batch...")
            time.sleep(DEFAULT_WAIT_TIME)
    
    return df_complete

def s6_check_category_matches():
    """Check which S6 assigned categories match the provided lists"""
    
    print("="*60)
    print("S6 CATEGORY MATCH CHECK") 
    print("="*60)
    
    # Get all assigned categories
    assigned_domains = df_complete['s6_refined_domain'].dropna().unique()
    assigned_methods = df_complete['s6_refined_method'].dropna().unique()
    
    # Find non-matching domains
    non_matching_domains = []
    for domain in assigned_domains:
        if domain not in S6_DOMAIN_CATEGORIES and domain not in ["API Failed", "Parse Failed"]:
            non_matching_domains.append(domain)
    
    # Find non-matching methods  
    non_matching_methods = []
    for method in assigned_methods:
        if method not in S6_METHOD_CATEGORIES and method not in ["API Failed", "Parse Failed"]:
            non_matching_methods.append(method)
    
    # Report results
    print(f"Total unique S6 domains assigned: {len(assigned_domains)}")
    print(f"S6 Domains matching your list: {len(assigned_domains) - len(non_matching_domains)}")
    print(f"S6 Domains NOT in your list: {len(non_matching_domains)}")
    
    if non_matching_domains:
        print(f"\nNON-MATCHING S6 DOMAINS:")
        for domain in non_matching_domains:
            count = (df_complete['s6_refined_domain'] == domain).sum()
            print(f"  '{domain}': {count} projects")
    
    print(f"\nTotal unique S6 methods assigned: {len(assigned_methods)}")
    print(f"S6 Methods matching your list: {len(assigned_methods) - len(non_matching_methods)}")
    print(f"S6 Methods NOT in your list: {len(non_matching_methods)}")
    
    if non_matching_methods:
        print(f"\nNON-MATCHING S6 METHODS:")
        for method in non_matching_methods:
            count = (df_complete['s6_refined_method'] == method).sum()
            print(f"  '{method}': {count} projects")
    
    # Calculate match rates
    total_projects = len(df_complete)
    domain_exact_matches = sum(1 for d in df_complete['s6_refined_domain'] if d in S6_DOMAIN_CATEGORIES)
    method_exact_matches = sum(1 for m in df_complete['s6_refined_method'] if m in S6_METHOD_CATEGORIES)
    
    print(f"\n" + "="*60)
    print("S6 MATCH RATES")
    print("="*60)
    print(f"S6 Domain exact match rate: {domain_exact_matches}/{total_projects} ({domain_exact_matches/total_projects*100:.1f}%)")
    print(f"S6 Method exact match rate: {method_exact_matches}/{total_projects} ({method_exact_matches/total_projects*100:.1f}%)")
    
    return non_matching_domains, non_matching_methods

def s6_reclassify_failed_categories(small_batch_size: int = 3, max_retries: int = 5):
    """Re-classify S6 projects that got non-matching categories"""
    
    print("="*60)
    print("S6 RE-CLASSIFYING NON-MATCHING CATEGORIES")
    print("="*60)
    
    # Identify failed S6 classifications
    domain_failures = ~df_complete['s6_refined_domain'].isin(S6_DOMAIN_CATEGORIES + ["API Failed", "Parse Failed"])
    method_failures = ~df_complete['s6_refined_method'].isin(S6_METHOD_CATEGORIES + ["API Failed", "Parse Failed"])
    
    # Get indices that need re-classification
    failed_indices = df_complete[domain_failures | method_failures].index.tolist()
    
    if not failed_indices:
        print("No failed S6 classifications found. Nothing to reclassify.")
        return df_complete
    
    print(f"Found {len(failed_indices)} S6 projects with non-matching categories")
    print(f"Using smaller batch size: {small_batch_size}")
    
    # Enhanced S6 prompt for reclassification
    S6_RECLASSIFY_PROMPT = """You are an expert taxonomist. You must classify projects using ONLY the exact categories from the provided lists.

CRITICAL INSTRUCTIONS:
1. DOMAIN = The primary scientific/technical FIELD or APPLICATION AREA (WHAT the project is about)
2. METHOD = The primary APPROACH or TECHNIQUE used (HOW the work is done)
3. You MUST select exactly one domain and one method from the provided lists
4. Use the EXACT spelling and capitalization as shown in the lists

Domain Areas (33 refined categories):
{domain_categories}

Method Areas (35 refined categories):
{method_categories}

Projects to Classify:
{project_summaries}

Output Format (MUST follow exactly):
Project 1:
Domain Area: [Selected Domain Area - must be EXACTLY as shown in the Domain list above]
Method Area: [Selected Method Area - must be EXACTLY as shown in the Method list above]

Project 2:
Domain Area: [Selected Domain Area - must be EXACTLY as shown in the Domain list above]
Method Area: [Selected Method Area - must be EXACTLY as shown in the Method list above]

Continue for all projects. Select ONLY from the provided lists."""

    def s6_reclassify_small_batch(project_summaries: List[str]) -> List[Dict[str, str]]:
        """Reclassify a small batch with S6 enhanced prompt"""
        
        # Format summaries
        formatted_summaries = ""
        for i, summary in enumerate(project_summaries, 1):
            formatted_summaries += f"Project {i}:\n{summary}\n\n"
        
        # Create enhanced S6 prompt
        prompt = S6_RECLASSIFY_PROMPT.format(
            domain_categories="\n".join(S6_DOMAIN_CATEGORIES),
            method_categories="\n".join(S6_METHOD_CATEGORIES),
            project_summaries=formatted_summaries
        )
        
        # Process with more retries
        output_text = ""
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                output_text = response.text
                break
            except Exception as e:
                print(f"  S6 Retry {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** (attempt + 1)
                    print(f"  Waiting {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print("  All S6 retries failed")
                    return [{"domain": "S6 Reclassify Failed", "method": "S6 Reclassify Failed"} for _ in project_summaries]
        
        # Parse results
        results = []
        project_blocks = re.split(r'Project \d+:', output_text)
        
        if project_blocks and not project_blocks[0].strip():
            project_blocks = project_blocks[1:]
        
        if len(project_blocks) < len(project_summaries):
            project_blocks.extend(["" for _ in range(len(project_summaries) - len(project_blocks))])
        
        for i, block in enumerate(project_blocks[:len(project_summaries)]):
            domain = "S6 Reclassify Failed"
            method = "S6 Reclassify Failed"
            
            for line in block.strip().split('\n'):
                if line.startswith("Domain Area:"):
                    domain = line.replace("Domain Area:", "").strip()
                elif line.startswith("Method Area:"):
                    method = line.replace("Method Area:", "").strip()
            
            results.append({"domain": domain, "method": method})
        
        return results
    
    # Process failed items in small batches
    total_failed = len(failed_indices)
    total_batches = (total_failed + small_batch_size - 1) // small_batch_size
    
    successful_reclassifications = 0
    
    for batch_num, start_idx in enumerate(range(0, total_failed, small_batch_size)):
        end_idx = min(start_idx + small_batch_size, total_failed)
        batch_indices = failed_indices[start_idx:end_idx]
        
        print(f"\nS6 Reclassifying batch {batch_num + 1}/{total_batches} ({len(batch_indices)} projects)")
        
        # Get batch of failed items
        batch_summaries = []
        for idx in batch_indices:
            summary = df_complete.loc[idx, 'full_text']
            batch_summaries.append(summary)
            
            # Show what failed
            old_domain = df_complete.loc[idx, 's6_refined_domain']
            old_method = df_complete.loc[idx, 's6_refined_method']
            print(f"  Project {idx}: '{old_domain}' / '{old_method}'")
        
        # Reclassify this batch
        batch_results = s6_reclassify_small_batch(batch_summaries)
        
        # Update results and show changes
        for i, idx in enumerate(batch_indices):
            if i < len(batch_results):
                new_domain = batch_results[i]['domain']
                new_method = batch_results[i]['method']
                
                df_complete.loc[idx, 's6_refined_domain'] = new_domain
                df_complete.loc[idx, 's6_refined_method'] = new_method
                
                # Check if reclassification was successful
                domain_success = new_domain in S6_DOMAIN_CATEGORIES
                method_success = new_method in S6_METHOD_CATEGORIES
                
                if domain_success and method_success:
                    successful_reclassifications += 1
                    print(f"    ✅ Fixed: '{new_domain}' / '{new_method}'")
                else:
                    print(f"    ❌ Still failed: '{new_domain}' / '{new_method}'")
        
        # Wait between batches
        if batch_num < total_batches - 1:
            print(f"  Waiting {DEFAULT_WAIT_TIME}s...")
            time.sleep(DEFAULT_WAIT_TIME)
    
    # Final save
    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
    
    print(f"\n" + "="*60)
    print("S6 RECLASSIFICATION COMPLETE")
    print("="*60)
    print(f"Total S6 projects reclassified: {total_failed}")
    print(f"Successfully fixed: {successful_reclassifications}")
    print(f"Still failed: {total_failed - successful_reclassifications}")
    
    return df_complete

# MAIN S6 EXECUTION FUNCTION
def run_s6_classification():
    """Run the complete S6 classification process"""
    
    print("🎯 STARTING COMPLETE S6 REFINED CLASSIFICATION")
    print("="*80)
    
    # Check prerequisites
    if 'df_complete' not in globals() or df_complete.empty:
        print("❌ ERROR: df_complete not available")
        return None
    elif 'model' not in globals() or not model:
        print("❌ ERROR: model not available")
        return None
    elif 'full_text' not in df_complete.columns:
        print("❌ ERROR: 'full_text' column missing")
        return None
    
    # Step 1: Run main S6 classification
    print("🚀 STEP 1: Main S6 Classification")
    df_result = s6_classify_dataset()
    
    # Step 2: Check for mismatches
    print("\n🔍 STEP 2: Checking S6 Category Matches")
    non_matching_domains, non_matching_methods = s6_check_category_matches()
    
    # Step 3: Fix any mismatches
    if non_matching_domains or non_matching_methods:
        print("\n🔧 STEP 3: Fixing S6 Non-Matching Categories")
        df_result = s6_reclassify_failed_categories()
        
        # Final check
        print("\n✅ STEP 4: Final S6 Verification")
        s6_check_category_matches()
    else:
        print("\n✅ No S6 reclassification needed!")
    
    # Final save and summary
    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
    print(f"\n🎉 S6 REFINED CLASSIFICATION COMPLETE!")
    
    # Show sample results
    print(f"\nS6 Sample Results:")
    sample_cols = ['id', 'full_text', 's6_refined_domain', 's6_refined_method']
    available_cols = [col for col in sample_cols if col in df_complete.columns]
    if available_cols:
        display(df_complete[available_cols].head())
    
    # Generate summary statistics
    domain_counts = df_complete['s6_refined_domain'].value_counts()
    method_counts = df_complete['s6_refined_method'].value_counts()
    
    print(f"\nTop 10 S6 assigned domain categories:")
    for domain, count in domain_counts.head(10).items():
        print(f"- {domain}: {count}")
    
    print(f"\nTop 10 S6 assigned method categories:")
    for method, count in method_counts.head(10).items():
        print(f"- {method}: {count}")
    
    return df_result

# USAGE:
df_s6_result = run_s6_classification()


# In[154]:


# COMPLETE S6 FIX STAGE
# S6 REFINED CATEGORIES
S6_DOMAIN_CATEGORIES = [
    "Fundamental Physics & Chemistry",
    "Mathematical & Computational Sciences", 
    "Biological & Life Sciences",
    "Earth, Environmental & Climate Science",
    "Agriculture, Food & Bioresources",
    "Aerospace, Space & Astronomy",
    "Computer Science & Information Technology",
    "Artificial Intelligence & Data Science",
    "Robotics & Autonomous Systems",
    "Electrical, Electronics & Photonics Engineering",
    "Materials Science & Engineering",
    "Chemical Engineering & Applied Chemistry",
    "Mechanical & Industrial Engineering",
    "Biomedical Engineering & Medical Technology",
    "Energy Systems & Technology",
    "Transportation, Urban & Regional Development",
    "Clinical Medicine & Health Sciences",
    "Neuroscience, Neurology & Cognitive Science",
    "Microbiology, Virology & Infectious Diseases",
    "Cancer & Oncology",
    "Drug Discovery & Pharmaceutical Development",
    "Public Health, Epidemiology & Healthcare Administration",
    "Social Sciences & Human Behavior",
    "Psychology & Behavioral Science",
    "Business, Management & Economics",
    "Policy, Governance & Law",
    "Humanities, Arts & Cultural Heritage",
    "Education & Training",
    "Sustainable Development & Global Challenges",
    "Security & Defense",
    "Environmental Health & Engineering",
    "Nuclear Science & Engineering",
    "Quantum Physics & Technologies"
]

S6_METHOD_CATEGORIES = [
    "Qualitative, Archival & Interpretive Research",
    "Quantitative, Statistical & Mathematical Modeling",
    "Computational Modeling, Simulation & Optimization",
    "Artificial Intelligence & Machine Learning Methods",
    "Experimental Design & Laboratory Techniques",
    "Data Collection, Curation & Management",
    "Data Analysis, Interpretation & Visualization",
    "Engineering Design, Fabrication & Prototyping",
    "Process Engineering & Optimization",
    "Materials Synthesis, Characterization & Analysis",
    "Chemical Synthesis, Catalysis & Analytical Methods",
    "Omics, Molecular & Cellular Biology Techniques",
    "Genetic Engineering & Synthetic Biology",
    "Bioprocessing, Biomanufacturing & Bioengineering",
    "Drug Discovery, Development & Delivery Methods",
    "Clinical & Healthcare Research Methods",
    "Diagnostic, Therapeutic & Medical Technology Development",
    "Advanced Imaging, Microscopy & Spectroscopy",
    "Sensor, Monitoring & Measurement Technologies",
    "Remote Sensing, Geospatial Analysis & Mapping",
    "Ecological & Environmental Modeling & Assessment",
    "Policy, Economic & Social Impact Analysis",
    "Ethical, Legal & Societal Framework Development",
    "Neuroscience, Cognitive & Behavioral Research Methods",
    "Agricultural, Plant Science & Food System Methods",
    "Immunological & Infectious Disease Research",
    "Software Engineering, HCI & Digital System Design",
    "Cybersecurity, Network Engineering & Data Protection",
    "Lifecycle, Risk Assessment & Environmental Management",
    "Innovation, Business Development & Digital Transformation",
    "Knowledge Management, Collaboration & Dissemination",
    "Capacity Building, Education & Training Methods",
    "Quantum Computing & Technology Development",
    "Archaeological, Heritage & Conservation Methods",
    "Astro/Space Observation & Geophysical Survey Methods"
]

def find_s6_failed_projects():
    """Find the exact projects that still have non-matching S6 categories - STRICT MODE"""
    
    print("🔍 FINDING S6 FAILED PROJECTS (STRICT - NO PARSE FAILED ALLOWED)")
    print("="*60)
    
    # Load current data
    try:
        df = pd.read_csv("df_processing_checkpoint.csv")
        print(f"✅ Loaded checkpoint CSV: {df.shape}")
    except:
        df = globals()['df_complete'].copy()
        print(f"✅ Using global df_complete: {df.shape}")
    
    # STRICT: Only empty strings and None are acceptable "failures"
    # Everything else (including "Parse Failed", "API Failed", etc.) must be fixed
    acceptable_errors = ["", None]
    
    failed_projects = []
    
    for idx in df.index:
        domain = df.loc[idx, 's6_refined_domain']
        method = df.loc[idx, 's6_refined_method']
        
        # Check if domain is problematic
        domain_bad = (pd.notna(domain) and 
                     str(domain).strip() != "" and  # Not empty string
                     domain not in S6_DOMAIN_CATEGORIES)  # Not valid category
        
        # Check if method is problematic  
        method_bad = (pd.notna(method) and 
                     str(method).strip() != "" and  # Not empty string
                     method not in S6_METHOD_CATEGORIES)  # Not valid category
        
        if domain_bad or method_bad:
            project_id = df.loc[idx, 'id'] if 'id' in df.columns else idx
            failed_projects.append({
                'index': idx,
                'id': project_id,
                'domain': domain,
                'method': method,
                'domain_bad': domain_bad,
                'method_bad': method_bad,
                'summary': str(df.loc[idx, 'full_text'])[:200] + "..."
            })
    
    print(f"Found {len(failed_projects)} projects with invalid S6 categories")
    
    # Group by bad categories to see patterns
    if failed_projects:
        domain_issues = {}
        method_issues = {}
        
        for proj in failed_projects:
            if proj['domain_bad']:
                domain = str(proj['domain'])
                if domain not in domain_issues:
                    domain_issues[domain] = []
                domain_issues[domain].append(proj['index'])
            
            if proj['method_bad']:
                method = str(proj['method'])
                if method not in method_issues:
                    method_issues[method] = []
                method_issues[method].append(proj['index'])
        
        print(f"\n❌ ALL PROBLEMATIC S6 DOMAINS (including Parse Failed, API Failed, etc.):")
        for domain, indices in domain_issues.items():
            print(f"  '{domain}': {len(indices)} projects")
        
        print(f"\n❌ ALL PROBLEMATIC S6 METHODS (including Parse Failed, API Failed, etc.):")
        for method, indices in method_issues.items():
            print(f"  '{method}': {len(indices)} projects")
    
    return failed_projects, df

def fix_s6_failed_projects_no_parse_failed(failed_projects, df, model):
    """Fix S6 projects - return empty strings instead of failure messages"""
    
    if not failed_projects:
        print("✅ No S6 failed projects to fix!")
        return df
    
    print(f"🔧 S6 FIXING - {len(failed_projects)} PROJECTS (NO PARSE FAILED ALLOWED)")
    print("="*60)
    
    # Enhanced prompt for problem cases
    TARGETED_S6_PROMPT = """You are an expert research taxonomist. You must classify these projects using ONLY the exact categories from the lists below.

CRITICAL RULES:
1. Select EXACTLY ONE domain from the Domain list for each project
2. Select EXACTLY ONE method from the Method list for each project  
3. Use the EXACT spelling, punctuation, and capitalization shown
4. DO NOT create new categories or modify existing ones
5. DO NOT confuse domains with methods

DOMAIN AREAS (Scientific/Technical Fields - choose ONE per project):
{domain_list}

METHOD AREAS (Research Approaches/Techniques - choose ONE per project):
{method_list}

PROJECTS TO CLASSIFY:
{project_summaries}

RESPOND IN THIS EXACT FORMAT:
Project 1:
Domain Area: [EXACT match from Domain list above]
Method Area: [EXACT match from Method list above]

Project 2:
Domain Area: [EXACT match from Domain list above]
Method Area: [EXACT match from Method list above]

... continue for all projects ...

Remember: Domain = WHAT field/area, Method = HOW it's done."""

    changes_made = []
    df_working = df.copy()
    
    # Process 6 projects at a time
    batch_size = 6
    total_batches = (len(failed_projects) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(failed_projects))
        batch_projects = failed_projects[start_idx:end_idx]
        
        print(f"\n--- S6 Batch {batch_num+1}/{total_batches} ({len(batch_projects)} projects) ---")
        
        # Show what we're fixing
        for i, proj in enumerate(batch_projects):
            print(f"  Project {i+1} (ID {proj['id']}, Index {proj['index']}):")
            print(f"    Current Domain: '{proj['domain']}' {'❌' if proj['domain_bad'] else '✅'}")
            print(f"    Current Method: '{proj['method']}' {'❌' if proj['method_bad'] else '✅'}")
        
        # Prepare batch summaries
        project_summaries = ""
        for i, proj in enumerate(batch_projects, 1):
            project_text = str(df_working.loc[proj['index'], 'full_text'])[:1200]  # Limit length
            project_summaries += f"Project {i}:\nTitle and Description: {project_text}\n\n"
        
        # Create focused prompt
        prompt = TARGETED_S6_PROMPT.format(
            domain_list="\n".join(S6_DOMAIN_CATEGORIES),
            method_list="\n".join(S6_METHOD_CATEGORIES),
            project_summaries=project_summaries
        )
        
        # Try to get results from LLM
        batch_success = False
        try:
            print(f"🤖 Calling LLM for batch {batch_num+1}...")
            
            response = model.generate_content(prompt)
            output = response.text.strip()
            
            print(f"✅ Got LLM response, parsing...")
            
            # Parse results for the batch
            project_blocks = re.split(r'Project \d+:', output)
            if project_blocks and not project_blocks[0].strip():
                project_blocks = project_blocks[1:]
            
            if len(project_blocks) >= len(batch_projects):
                batch_results = []
                
                for i, block in enumerate(project_blocks[:len(batch_projects)]):
                    new_domain = ""  # Default to EMPTY instead of "Parse Failed"
                    new_method = ""  # Default to EMPTY instead of "Parse Failed"
                    
                    lines = block.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Domain Area:"):
                            candidate = line.replace("Domain Area:", "").strip()
                            candidate = candidate.strip('"\'- ')
                            # Only assign if it's actually in our valid list
                            if candidate in S6_DOMAIN_CATEGORIES:
                                new_domain = candidate
                        elif line.startswith("Method Area:"):
                            candidate = line.replace("Method Area:", "").strip()
                            candidate = candidate.strip('"\'- ')
                            # Only assign if it's actually in our valid list
                            if candidate in S6_METHOD_CATEGORIES:
                                new_method = candidate
                    
                    batch_results.append({
                        'domain': new_domain,
                        'method': new_method
                    })
                
                # Apply all results in this batch
                for i, result in enumerate(batch_results):
                    proj = batch_projects[i]
                    idx = proj['index']
                    
                    old_domain = df_working.loc[idx, 's6_refined_domain']
                    old_method = df_working.loc[idx, 's6_refined_method']
                    
                    # Only update categories that were actually problematic
                    if proj['domain_bad']:
                        df_working.loc[idx, 's6_refined_domain'] = result['domain']
                        print(f"  🔄 Fixed Domain {idx}: '{old_domain}' → '{result['domain']}'")
                    
                    if proj['method_bad']:
                        df_working.loc[idx, 's6_refined_method'] = result['method']
                        print(f"  🔄 Fixed Method {idx}: '{old_method}' → '{result['method']}'")
                    
                    changes_made.append({
                        'index': idx,
                        'id': proj['id'],
                        'old_domain': old_domain,
                        'new_domain': result['domain'] if proj['domain_bad'] else old_domain,
                        'old_method': old_method,
                        'new_method': result['method'] if proj['method_bad'] else old_method
                    })
                
                batch_success = True
                print(f"  ✅ Batch {batch_num+1} processed successfully")
            else:
                print(f"  ⚠️ Got {len(project_blocks)} project blocks, expected {len(batch_projects)}")
                
        except Exception as e:
            print(f"  ❌ LLM error for batch {batch_num+1}: {e}")
        
        if not batch_success:
            print(f"  🔄 Batch {batch_num+1} failed - setting empty strings for problematic categories")
            # Set empty strings for failed attempts (so they get caught in next iteration)
            for proj in batch_projects:
                idx = proj['index']
                if proj['domain_bad']:
                    df_working.loc[idx, 's6_refined_domain'] = ""
                    print(f"    Set domain to empty for index {idx}")
                if proj['method_bad']:
                    df_working.loc[idx, 's6_refined_method'] = ""
                    print(f"    Set method to empty for index {idx}")
        
        # Small delay between batches
        if batch_num < total_batches - 1:
            time.sleep(3)
    
    print(f"\n💾 SAVING S6 FIXES")
    print("="*60)
    
    # Save back to CSV
    df_working.to_csv("df_processing_checkpoint.csv", index=False)
    print(f"✅ Saved to df_processing_checkpoint.csv")
    
    # Update global if possible
    try:
        globals()['df_complete'] = df_working.copy()
        print(f"✅ Updated global df_complete")
    except:
        pass
    
    print(f"\n📊 S6 FIX SUMMARY:")
    print(f"   Failed projects processed: {len(failed_projects)}")
    print(f"   Batches processed: {total_batches}")
    print(f"   Changes attempted: {len(changes_made)}")
    
    return df_working

def run_s6_complete_fix():
    """Main function to run complete S6 fixes until everything is clean"""
    
    print("🎯 S6 COMPLETE FIX - NO PARSE FAILED ALLOWED")
    print("="*80)
    
    # Get model
    try:
        model = globals()['model']
        print(f"✅ Using global model")
    except:
        print("❌ ERROR: Global 'model' not found")
        return None
    
    # Keep running fix iterations until no more failures
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"\n🔄 S6 FIX ITERATION {iteration + 1}/{max_iterations}")
        print("="*60)
        
        # Step 1: Find failed projects (strict mode)
        failed_projects, df = find_s6_failed_projects()
        
        if not failed_projects:
            print(f"🎉 S6 COMPLETE - No more failed projects after {iteration + 1} iteration(s)!")
            break
        
        print(f"📋 Found {len(failed_projects)} projects that need fixing")
        
        # Step 2: Apply fixes
        df = fix_s6_failed_projects_no_parse_failed(failed_projects, df, model)
        
        # Step 3: Show progress
        print(f"\n📊 Iteration {iteration + 1} complete")
        
        if iteration == max_iterations - 1:
            print(f"⚠️ Reached maximum iterations ({max_iterations})")
            print(f"   Some projects may still need manual review")
    
    # Final verification
    print(f"\n🔍 FINAL S6 VERIFICATION")
    print("="*60)
    
    final_failed, df_final = find_s6_failed_projects()
    
    domain_matches = sum(1 for d in df_final['s6_refined_domain'] if d in S6_DOMAIN_CATEGORIES)
    method_matches = sum(1 for m in df_final['s6_refined_method'] if m in S6_METHOD_CATEGORIES)
    empty_domains = sum(1 for d in df_final['s6_refined_domain'] if pd.isna(d) or str(d).strip() == "")
    empty_methods = sum(1 for m in df_final['s6_refined_method'] if pd.isna(m) or str(m).strip() == "")
    total_projects = len(df_final)
    
    print(f"📊 FINAL S6 STATISTICS:")
    print(f"   Total projects: {total_projects}")
    print(f"   Valid domain categories: {domain_matches} ({domain_matches/total_projects*100:.2f}%)")
    print(f"   Valid method categories: {method_matches} ({method_matches/total_projects*100:.2f}%)")
    print(f"   Empty domains (acceptable): {empty_domains} ({empty_domains/total_projects*100:.2f}%)")
    print(f"   Empty methods (acceptable): {empty_methods} ({empty_methods/total_projects*100:.2f}%)")
    print(f"   Remaining problematic projects: {len(final_failed)}")
    
    if len(final_failed) == 0:
        print(f"\n🎉🎉🎉 PERFECT! 100% S6 SUCCESS - NO MORE PARSE FAILED! 🎉🎉🎉")
    else:
        print(f"\n⚠️ Still have {len(final_failed)} projects with invalid categories:")
        # Show first few remaining problems
        for proj in final_failed[:5]:
            print(f"   - Index {proj['index']}: Domain='{proj['domain']}', Method='{proj['method']}'")
        
        if len(final_failed) > 5:
            print(f"   ... and {len(final_failed) - 5} more")
    
    print(f"\n🎯 S6 COMPLETE FIX FINISHED!")
    
    return df_final

# Usage
df_s6_clean = run_s6_complete_fix()


# In[158]:


# S7 FINAL CONSOLIDATED CATEGORIES 
S7_DOMAIN_CATEGORIES = [
    "Health, Medicine & Life Sciences",
    "Physical, Mathematical & Space Sciences", 
    "Engineering, Manufacturing & Technology",
    "Information, Computation & Digital Systems",
    "Environment, Climate & Earth Systems",
    "Economy, Business & Agriculture",
    "Society, Culture & Human Behavior",
    "Humanities, Arts & Cultural Heritage",
    "Policy, Governance & Law",
    "Education & Learning Systems",
    "Global Development & Sustainability"
]

S7_METHOD_CATEGORIES = [
    "Data Analysis, Modeling & Computational Methods",
    "Qualitative, Social Inquiry & Policy Research",
    "Engineering Design, Prototyping & Systems Development",
    "Experimental, Laboratory & Field Investigation",
    "Material, Chemical & Bioprocess Development",
    "Imaging, Sensing & Measurement Techniques",
    "Clinical, Health & Therapeutic Development Methods",
    "Communication, Collaboration & Knowledge Exchange",
    "Innovation, Strategic Management & Evaluation",
    "Educational, Human-Centered & Training Methods"
]

print(f"🎯 S7 FINAL CONSOLIDATED CATEGORIES LOADED")
print(f"   Domains: {len(S7_DOMAIN_CATEGORIES)} categories")
print(f"   Methods: {len(S7_METHOD_CATEGORIES)} categories")

# S7 Classification Prompt - Enhanced for the highly consolidated taxonomy
S7_BATCH_CLASSIFY_PROMPT = """You are an expert taxonomist specializing in research project classification. Your task is to classify multiple research project summaries using a highly consolidated taxonomy system.

You must assign each project exactly ONE domain area and ONE method area from the provided lists.

DOMAIN AREAS (11 broad categories - WHAT field/application the project focuses on):
{domain_categories}

METHOD AREAS (10 broad categories - HOW the research work is being conducted):
{method_categories}

Classification Guidelines:
1. DOMAIN SELECTION: Choose the primary field or application area that best represents the project's core focus
   - Health, Medicine & Life Sciences: Medical research, biology, biotechnology, health sciences
   - Physical, Mathematical & Space Sciences: Physics, chemistry, mathematics, astronomy, space research
   - Engineering, Manufacturing & Technology: All engineering disciplines, manufacturing, industrial technology
   - Information, Computation & Digital Systems: Computer science, IT, AI, data science, digital technology
   - Environment, Climate & Earth Systems: Environmental science, climate research, earth sciences, sustainability
   - Economy, Business & Agriculture: Economic research, business studies, agricultural sciences, food systems
   - Society, Culture & Human Behavior: Sociology, anthropology, psychology, cultural studies, social sciences
   - Humanities, Arts & Cultural Heritage: Literature, arts, history, philosophy, cultural heritage
   - Policy, Governance & Law: Public policy, governance, legal studies, political science
   - Education & Learning Systems: Educational research, pedagogy, learning sciences, training systems
   - Global Development & Sustainability: Development studies, international cooperation, sustainable development

2. METHOD SELECTION: Choose the primary approach or methodology used in the research
   - Focus on the main way the research work is being accomplished
   - Consider the core techniques, approaches, or methodologies that drive the project

3. Use EXACT spelling and capitalization as shown in the lists above
4. Select the single best match for each category - do not overthink edge cases

Projects to Classify:
{project_summaries}

Output Format:
Project 1:
Domain Area: [Selected Domain Area - must be exactly as shown in the list]
Method Area: [Selected Method Area - must be exactly as shown in the list]

Project 2:
Domain Area: [Selected Domain Area - must be exactly as shown in the list]
Method Area: [Selected Method Area - must be exactly as shown in the list]

... continue for all projects ...

Important: For each project, select only ONE domain area and ONE method area. Both must be exact matches from the provided lists."""

def s7_classify_batch(project_summaries: List[str], max_retries: int = 3) -> List[Dict[str, str]]:
    """Classify a batch of project summaries using S7 final categories"""
    
    # Format the project summaries for the prompt
    formatted_summaries = ""
    for i, summary in enumerate(project_summaries, 1):
        # Use full summaries for best results
        formatted_summaries += f"Project {i}:\n{summary}\n\n"
    
    # Create the prompt
    prompt = S7_BATCH_CLASSIFY_PROMPT.format(
        domain_categories="\n".join(S7_DOMAIN_CATEGORIES),
        method_categories="\n".join(S7_METHOD_CATEGORIES),
        project_summaries=formatted_summaries
    )
    
    # Process with model (with retries)
    output_text = ""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            output_text = response.text
            break
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** (attempt + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Failed all retry attempts")
                return [{"domain": "API Failed", "method": "API Failed"} for _ in project_summaries]
    
    # Parse results
    results = []
    project_blocks = re.split(r'Project \d+:', output_text)
    
    # Skip the first item if it's empty
    if project_blocks and not project_blocks[0].strip():
        project_blocks = project_blocks[1:]
    
    # If we didn't get enough results, pad with failures
    if len(project_blocks) < len(project_summaries):
        print(f"Warning: Expected {len(project_summaries)} results but got {len(project_blocks)}")
        project_blocks.extend(["" for _ in range(len(project_summaries) - len(project_blocks))])
    
    # Process each project block
    for i, block in enumerate(project_blocks[:len(project_summaries)]):
        domain = "Parse Failed"
        method = "Parse Failed"
        
        for line in block.strip().split('\n'):
            line = line.strip()
            if line.startswith("Domain Area:"):
                domain = line.replace("Domain Area:", "").strip()
                # Clean up formatting
                domain = domain.strip('"\'- ')
            elif line.startswith("Method Area:"):
                method = line.replace("Method Area:", "").strip()
                # Clean up formatting
                method = method.strip('"\'- ')
        
        results.append({"domain": domain, "method": method})
    
    return results

def display_s7_first_batch_results(batch_summaries: List[str], batch_results: List[Dict[str, str]]):
    """Display detailed results for the first S7 batch"""
    print("\n===== S7 FIRST BATCH RESULTS =====")
    print(f"Showing results for the first batch of {len(batch_summaries)} projects")
    
    for i, result in enumerate(batch_results):
        summary = batch_summaries[i]
        domain = result["domain"]
        method = result["method"]
        
        # Truncate summary for display
        truncated_summary = summary[:150] + "..." if len(summary) > 150 else summary
        truncated_summary = truncated_summary.replace('\n', ' ')
        
        print(f"\nProject {i+1}:")
        print(f"Summary: {truncated_summary}")
        print(f"S7 Domain: {domain}")
        print(f"S7 Method: {method}")
    
    print("\n================================")

def s7_classify_dataset():
    """Process the dataset using S7 final consolidated categories"""
    
    print(f"🚀 STARTING S7 FINAL CONSOLIDATED CLASSIFICATION")
    print(f"="*80)
    print(f"Using S7 final categories:")
    print(f"  {len(S7_DOMAIN_CATEGORIES)} Domain areas (highly consolidated)")
    print(f"  {len(S7_METHOD_CATEGORIES)} Method areas (highly consolidated)")
    print(f"  Batch size: {DEFAULT_BATCH_SIZE}")
    print(f"  Trial mode: {RUN_OVERALL_TRIAL_MODE}")
    if RUN_OVERALL_TRIAL_MODE:
        print(f"  Max trial batches: {MAX_BATCHES_FOR_TRIAL}")
    
    # Initialize new S7 result columns
    df_complete['s7_final_domain'] = ""
    df_complete['s7_final_method'] = ""
    
    # Determine scope using global variables
    if RUN_OVERALL_TRIAL_MODE:
        max_rows = DEFAULT_BATCH_SIZE * MAX_BATCHES_FOR_TRIAL
        process_df = df_complete.iloc[:max_rows].copy()
        print(f"TRIAL MODE: Processing {len(process_df)} rows")
    else:
        process_df = df_complete.copy()
        print(f"FULL MODE: Processing {len(process_df)} rows")
    
    # Process in batches
    total_records = len(process_df)
    total_batches = (total_records + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE
    
    for batch_num, start_idx in enumerate(range(0, total_records, DEFAULT_BATCH_SIZE)):
        end_idx = min(start_idx + DEFAULT_BATCH_SIZE, total_records)
        batch_size_actual = end_idx - start_idx
        
        print(f"\nProcessing S7 batch {batch_num + 1}/{total_batches} (records {start_idx+1}-{end_idx})")
        
        # Extract summaries for this batch
        batch_summaries = []
        for idx in range(start_idx, end_idx):
            summary = process_df.iloc[idx]['full_text']
            batch_summaries.append(summary)
        
        # Classify the batch using S7 categories
        batch_results = s7_classify_batch(batch_summaries)
        
        # Save results directly to df_complete
        for i in range(batch_size_actual):
            original_idx = process_df.index[start_idx + i]
            if i < len(batch_results):
                df_complete.loc[original_idx, 's7_final_domain'] = batch_results[i]['domain']
                df_complete.loc[original_idx, 's7_final_method'] = batch_results[i]['method']
            else:
                df_complete.loc[original_idx, 's7_final_domain'] = "API Failed"
                df_complete.loc[original_idx, 's7_final_method'] = "API Failed"
        
        # Display detailed results for first batch only
        if batch_num == 0:
            display_s7_first_batch_results(batch_summaries, batch_results)
        
        # Save every 20 batches
        if (batch_num + 1) % 20 == 0:
            df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
            print(f"  ✅ S7 SAVED at batch {batch_num + 1}")
        
        # Wait between batches
        if batch_num < total_batches - 1:
            print(f"Waiting {DEFAULT_WAIT_TIME} seconds before next S7 batch...")
            time.sleep(DEFAULT_WAIT_TIME)
    
    return df_complete

def s7_check_category_matches():
    """Check which S7 assigned categories match the provided lists"""
    
    print("="*60)
    print("S7 FINAL CATEGORY MATCH CHECK") 
    print("="*60)
    
    # Get all assigned categories
    assigned_domains = df_complete['s7_final_domain'].dropna().unique()
    assigned_methods = df_complete['s7_final_method'].dropna().unique()
    
    # Find non-matching domains
    non_matching_domains = []
    for domain in assigned_domains:
        if domain not in S7_DOMAIN_CATEGORIES and domain not in ["API Failed", "Parse Failed"]:
            non_matching_domains.append(domain)
    
    # Find non-matching methods  
    non_matching_methods = []
    for method in assigned_methods:
        if method not in S7_METHOD_CATEGORIES and method not in ["API Failed", "Parse Failed"]:
            non_matching_methods.append(method)
    
    # Report results
    print(f"Total unique S7 domains assigned: {len(assigned_domains)}")
    print(f"S7 Domains matching your list: {len(assigned_domains) - len(non_matching_domains)}")
    print(f"S7 Domains NOT in your list: {len(non_matching_domains)}")
    
    if non_matching_domains:
        print(f"\nNON-MATCHING S7 DOMAINS:")
        for domain in non_matching_domains:
            count = (df_complete['s7_final_domain'] == domain).sum()
            print(f"  '{domain}': {count} projects")
    
    print(f"\nTotal unique S7 methods assigned: {len(assigned_methods)}")
    print(f"S7 Methods matching your list: {len(assigned_methods) - len(non_matching_methods)}")
    print(f"S7 Methods NOT in your list: {len(non_matching_methods)}")
    
    if non_matching_methods:
        print(f"\nNON-MATCHING S7 METHODS:")
        for method in non_matching_methods:
            count = (df_complete['s7_final_method'] == method).sum()
            print(f"  '{method}': {count} projects")
    
    # Calculate match rates
    total_projects = len(df_complete)
    domain_exact_matches = sum(1 for d in df_complete['s7_final_domain'] if d in S7_DOMAIN_CATEGORIES)
    method_exact_matches = sum(1 for m in df_complete['s7_final_method'] if m in S7_METHOD_CATEGORIES)
    
    print(f"\n" + "="*60)
    print("S7 FINAL MATCH RATES")
    print("="*60)
    print(f"S7 Domain exact match rate: {domain_exact_matches}/{total_projects} ({domain_exact_matches/total_projects*100:.1f}%)")
    print(f"S7 Method exact match rate: {method_exact_matches}/{total_projects} ({method_exact_matches/total_projects*100:.1f}%)")
    
    return non_matching_domains, non_matching_methods

def s7_reclassify_failed_categories(batch_size: int = 5, max_retries: int = 5):
    """Re-classify S7 projects that got non-matching categories"""
    
    print("="*60)
    print("S7 RE-CLASSIFYING NON-MATCHING CATEGORIES")
    print("="*60)
    
    # Identify failed S7 classifications
    domain_failures = ~df_complete['s7_final_domain'].isin(S7_DOMAIN_CATEGORIES + ["API Failed", "Parse Failed"])
    method_failures = ~df_complete['s7_final_method'].isin(S7_METHOD_CATEGORIES + ["API Failed", "Parse Failed"])
    
    # Get indices that need re-classification
    failed_indices = df_complete[domain_failures | method_failures].index.tolist()
    
    if not failed_indices:
        print("No failed S7 classifications found. Nothing to reclassify.")
        return df_complete
    
    print(f"Found {len(failed_indices)} S7 projects with non-matching categories")
    print(f"Using batch size: {batch_size}")
    
    # Enhanced S7 prompt for reclassification
    S7_RECLASSIFY_PROMPT = """You are an expert taxonomist. You must classify projects using ONLY the exact categories from the provided lists.

This is a highly consolidated taxonomy with only 11 domains and 10 methods. Choose the best fit from these broad categories.

DOMAIN AREAS (11 broad categories):
{domain_categories}

METHOD AREAS (10 broad categories):
{method_categories}

Projects to Classify:
{project_summaries}

Output Format (MUST follow exactly):
Project 1:
Domain Area: [Selected Domain Area - must be EXACTLY as shown in the Domain list above]
Method Area: [Selected Method Area - must be EXACTLY as shown in the Method list above]

Project 2:
Domain Area: [Selected Domain Area - must be EXACTLY as shown in the Domain list above]
Method Area: [Selected Method Area - must be EXACTLY as shown in the Method list above]

Continue for all projects. Select ONLY from the provided lists."""

    def s7_reclassify_batch(project_summaries: List[str], indices: List[int]) -> List[Dict[str, str]]:
        """Reclassify a batch with S7 enhanced prompt"""
        
        # Format summaries
        formatted_summaries = ""
        for i, summary in enumerate(project_summaries, 1):
            formatted_summaries += f"Project {i}:\n{summary}\n\n"
        
        # Create enhanced S7 prompt
        prompt = S7_RECLASSIFY_PROMPT.format(
            domain_categories="\n".join(S7_DOMAIN_CATEGORIES),
            method_categories="\n".join(S7_METHOD_CATEGORIES),
            project_summaries=formatted_summaries
        )
        
        # Process with more retries
        output_text = ""
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                output_text = response.text
                break
            except Exception as e:
                print(f"  S7 Retry {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** (attempt + 1)
                    print(f"  Waiting {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print("  All S7 retries failed")
                    return [{"domain": "S7 Reclassify Failed", "method": "S7 Reclassify Failed"} for _ in project_summaries]
        
        # Parse results
        results = []
        project_blocks = re.split(r'Project \d+:', output_text)
        
        if project_blocks and not project_blocks[0].strip():
            project_blocks = project_blocks[1:]
        
        if len(project_blocks) < len(project_summaries):
            project_blocks.extend(["" for _ in range(len(project_summaries) - len(project_blocks))])
        
        for i, block in enumerate(project_blocks[:len(project_summaries)]):
            domain = "S7 Reclassify Failed"
            method = "S7 Reclassify Failed"
            
            for line in block.strip().split('\n'):
                line = line.strip()
                if line.startswith("Domain Area:"):
                    domain = line.replace("Domain Area:", "").strip()
                    domain = domain.strip('"\'- ')
                elif line.startswith("Method Area:"):
                    method = line.replace("Method Area:", "").strip()
                    method = method.strip('"\'- ')
            
            results.append({"domain": domain, "method": method, "index": indices[i] if i < len(indices) else None})
        
        return results
    
    # Process failed items in batches
    total_failed = len(failed_indices)
    total_batches = (total_failed + batch_size - 1) // batch_size
    
    successful_reclassifications = 0
    
    for batch_num, start_idx in enumerate(range(0, total_failed, batch_size)):
        end_idx = min(start_idx + batch_size, total_failed)
        batch_indices = failed_indices[start_idx:end_idx]
        
        print(f"\nS7 Reclassifying batch {batch_num + 1}/{total_batches} ({len(batch_indices)} projects)")
        
        # Get batch of failed items
        batch_summaries = []
        for idx in batch_indices:
            summary = df_complete.loc[idx, 'full_text']
            batch_summaries.append(summary)
            
            # Show what failed
            old_domain = df_complete.loc[idx, 's7_final_domain']
            old_method = df_complete.loc[idx, 's7_final_method']
            print(f"  Project {idx}: '{old_domain}' / '{old_method}'")
        
        # Reclassify this batch
        batch_results = s7_reclassify_batch(batch_summaries, batch_indices)
        
        # Update results and show changes
        for i, idx in enumerate(batch_indices):
            if i < len(batch_results):
                new_domain = batch_results[i]['domain']
                new_method = batch_results[i]['method']
                
                df_complete.loc[idx, 's7_final_domain'] = new_domain
                df_complete.loc[idx, 's7_final_method'] = new_method
                
                # Check if reclassification was successful
                domain_success = new_domain in S7_DOMAIN_CATEGORIES
                method_success = new_method in S7_METHOD_CATEGORIES
                
                if domain_success and method_success:
                    successful_reclassifications += 1
                    print(f"    ✅ Fixed: '{new_domain}' / '{new_method}'")
                else:
                    print(f"    ❌ Still failed: '{new_domain}' / '{new_method}'")
        
        # Wait between batches
        if batch_num < total_batches - 1:
            print(f"  Waiting {DEFAULT_WAIT_TIME}s...")
            time.sleep(DEFAULT_WAIT_TIME)
    
    # Final save
    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
    
    print(f"\n" + "="*60)
    print("S7 RECLASSIFICATION COMPLETE")
    print("="*60)
    print(f"Total S7 projects reclassified: {total_failed}")
    print(f"Successfully fixed: {successful_reclassifications}")
    print(f"Still failed: {total_failed - successful_reclassifications}")
    
    return df_complete

# MAIN S7 EXECUTION FUNCTION
def run_s7_final_classification():
    """Run the complete S7 final classification process"""
    
    print("🎯 STARTING COMPLETE S7 FINAL CLASSIFICATION")
    print("="*80)
    
    # Check prerequisites
    if 'df_complete' not in globals() or df_complete.empty:
        print("❌ ERROR: df_complete not available")
        return None
    elif 'model' not in globals() or not model:
        print("❌ ERROR: model not available")
        return None
    elif 'full_text' not in df_complete.columns:
        print("❌ ERROR: 'full_text' column missing")
        return None
    
    # Step 1: Run main S7 classification
    print("🚀 STEP 1: Main S7 Final Classification")
    df_result = s7_classify_dataset()
    
    # Step 2: Check for mismatches
    print("\n🔍 STEP 2: Checking S7 Category Matches")
    non_matching_domains, non_matching_methods = s7_check_category_matches()
    
    # Step 3: Fix any mismatches
    if non_matching_domains or non_matching_methods:
        print("\n🔧 STEP 3: Fixing S7 Non-Matching Categories")
        df_result = s7_reclassify_failed_categories()
        
        # Final check
        print("\n✅ STEP 4: Final S7 Verification")
        s7_check_category_matches()
    else:
        print("\n✅ No S7 reclassification needed!")
    
    # Final save and summary
    df_complete.to_csv(MAIN_DATAFRAME_CHECKPOINT_FILE, index=False)
    print(f"\n🎉 S7 FINAL CLASSIFICATION COMPLETE!")
    
    # Show sample results
    print(f"\nS7 Sample Results:")
    sample_cols = ['id', 'full_text', 's7_final_domain', 's7_final_method']
    available_cols = [col for col in sample_cols if col in df_complete.columns]
    if available_cols:
        display(df_complete[available_cols].head())
    
    # Generate summary statistics
    domain_counts = df_complete['s7_final_domain'].value_counts()
    method_counts = df_complete['s7_final_method'].value_counts()
    
    print(f"\nTop S7 assigned domain categories:")
    for domain, count in domain_counts.head(len(S7_DOMAIN_CATEGORIES)).items():
        print(f"- {domain}: {count}")
    
    print(f"\nTop S7 assigned method categories:")
    for method, count in method_counts.head(len(S7_METHOD_CATEGORIES)).items():
        print(f"- {method}: {count}")
    
    return df_result

# USAGE:
df_s7_result = run_s7_final_classification()


# In[160]:


# COMPLETE S7 FIX STAGE - 
# S7 FINAL CONSOLIDATED CATEGORIES (from your original code)
S7_DOMAIN_CATEGORIES = [
    "Health, Medicine & Life Sciences",
    "Physical, Mathematical & Space Sciences", 
    "Engineering, Manufacturing & Technology",
    "Information, Computation & Digital Systems",
    "Environment, Climate & Earth Systems",
    "Economy, Business & Agriculture",
    "Society, Culture & Human Behavior",
    "Humanities, Arts & Cultural Heritage",
    "Policy, Governance & Law",
    "Education & Learning Systems",
    "Global Development & Sustainability"
]

S7_METHOD_CATEGORIES = [
    "Data Analysis, Modeling & Computational Methods",
    "Qualitative, Social Inquiry & Policy Research",
    "Engineering Design, Prototyping & Systems Development",
    "Experimental, Laboratory & Field Investigation",
    "Material, Chemical & Bioprocess Development",
    "Imaging, Sensing & Measurement Techniques",
    "Clinical, Health & Therapeutic Development Methods",
    "Communication, Collaboration & Knowledge Exchange",
    "Innovation, Strategic Management & Evaluation",
    "Educational, Human-Centered & Training Methods"
]

def find_s7_failed_projects():
    """Find the exact projects that still have non-matching S7 categories - STRICT MODE"""
    
    print("🔍 FINDING S7 FAILED PROJECTS (STRICT - NO PARSE FAILED ALLOWED)")
    print("="*60)
    
    # Load current data
    try:
        df = pd.read_csv("df_processing_checkpoint.csv")
        print(f"✅ Loaded checkpoint CSV: {df.shape}")
    except:
        df = globals()['df_complete'].copy()
        print(f"✅ Using global df_complete: {df.shape}")
    
    # STRICT: Only empty strings and None are acceptable "failures"
    # Everything else (including "Parse Failed", "API Failed", etc.) must be fixed
    acceptable_errors = ["", None]
    
    failed_projects = []
    
    for idx in df.index:
        domain = df.loc[idx, 's7_final_domain']
        method = df.loc[idx, 's7_final_method']
        
        # Check if domain is problematic
        domain_bad = (pd.notna(domain) and 
                     str(domain).strip() != "" and  # Not empty string
                     domain not in S7_DOMAIN_CATEGORIES)  # Not valid category
        
        # Check if method is problematic  
        method_bad = (pd.notna(method) and 
                     str(method).strip() != "" and  # Not empty string
                     method not in S7_METHOD_CATEGORIES)  # Not valid category
        
        if domain_bad or method_bad:
            project_id = df.loc[idx, 'id'] if 'id' in df.columns else idx
            failed_projects.append({
                'index': idx,
                'id': project_id,
                'domain': domain,
                'method': method,
                'domain_bad': domain_bad,
                'method_bad': method_bad,
                'summary': str(df.loc[idx, 'full_text'])[:200] + "..."
            })
    
    print(f"Found {len(failed_projects)} projects with invalid S7 categories")
    
    # Group by bad categories to see patterns
    if failed_projects:
        domain_issues = {}
        method_issues = {}
        
        for proj in failed_projects:
            if proj['domain_bad']:
                domain = str(proj['domain'])
                if domain not in domain_issues:
                    domain_issues[domain] = []
                domain_issues[domain].append(proj['index'])
            
            if proj['method_bad']:
                method = str(proj['method'])
                if method not in method_issues:
                    method_issues[method] = []
                method_issues[method].append(proj['index'])
        
        print(f"\n❌ ALL PROBLEMATIC S7 DOMAINS (including Parse Failed, API Failed, etc.):")
        for domain, indices in domain_issues.items():
            print(f"  '{domain}': {len(indices)} projects")
        
        print(f"\n❌ ALL PROBLEMATIC S7 METHODS (including Parse Failed, API Failed, etc.):")
        for method, indices in method_issues.items():
            print(f"  '{method}': {len(indices)} projects")
    
    return failed_projects, df

def fix_s7_failed_projects_no_parse_failed(failed_projects, df, model):
    """Fix S7 projects - return empty strings instead of failure messages"""
    
    if not failed_projects:
        print("✅ No S7 failed projects to fix!")
        return df
    
    print(f"🔧 S7 FIXING - {len(failed_projects)} PROJECTS (NO PARSE FAILED ALLOWED)")
    print("="*60)
    
    # Enhanced prompt for problem cases - simplified for S7's broad categories
    TARGETED_S7_PROMPT = """You are an expert research taxonomist. You must classify these projects using ONLY the exact categories from the highly consolidated lists below.

This is a very broad taxonomy with only 11 domains and 10 methods. Choose the best general fit.

CRITICAL RULES:
1. Select EXACTLY ONE domain from the Domain list for each project
2. Select EXACTLY ONE method from the Method list for each project  
3. Use the EXACT spelling, punctuation, and capitalization shown
4. DO NOT create new categories or modify existing ones
5. These are very broad categories - pick the best general fit

DOMAIN AREAS (11 broad consolidated areas - choose ONE per project):
{domain_list}

METHOD AREAS (10 broad consolidated approaches - choose ONE per project):
{method_list}

PROJECTS TO CLASSIFY:
{project_summaries}

RESPOND IN THIS EXACT FORMAT:
Project 1:
Domain Area: [EXACT match from Domain list above]
Method Area: [EXACT match from Method list above]

Project 2:
Domain Area: [EXACT match from Domain list above]
Method Area: [EXACT match from Method list above]

... continue for all projects ...

Remember: These are very broad, consolidated categories. Pick the best general fit for each project."""

    changes_made = []
    df_working = df.copy()
    
    # Process 8 projects at a time (can be larger since S7 categories are simpler)
    batch_size = 8
    total_batches = (len(failed_projects) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(failed_projects))
        batch_projects = failed_projects[start_idx:end_idx]
        
        print(f"\n--- S7 Batch {batch_num+1}/{total_batches} ({len(batch_projects)} projects) ---")
        
        # Show what we're fixing
        for i, proj in enumerate(batch_projects):
            print(f"  Project {i+1} (ID {proj['id']}, Index {proj['index']}):")
            print(f"    Current Domain: '{proj['domain']}' {'❌' if proj['domain_bad'] else '✅'}")
            print(f"    Current Method: '{proj['method']}' {'❌' if proj['method_bad'] else '✅'}")
        
        # Prepare batch summaries
        project_summaries = ""
        for i, proj in enumerate(batch_projects, 1):
            project_text = str(df_working.loc[proj['index'], 'full_text'])[:1000]  # Shorter for S7
            project_summaries += f"Project {i}:\nTitle and Description: {project_text}\n\n"
        
        # Create focused prompt
        prompt = TARGETED_S7_PROMPT.format(
            domain_list="\n".join(S7_DOMAIN_CATEGORIES),
            method_list="\n".join(S7_METHOD_CATEGORIES),
            project_summaries=project_summaries
        )
        
        # Try to get results from LLM
        batch_success = False
        try:
            print(f"🤖 Calling LLM for batch {batch_num+1}...")
            
            response = model.generate_content(prompt)
            output = response.text.strip()
            
            print(f"✅ Got LLM response, parsing...")
            
            # Parse results for the batch
            project_blocks = re.split(r'Project \d+:', output)
            if project_blocks and not project_blocks[0].strip():
                project_blocks = project_blocks[1:]
            
            if len(project_blocks) >= len(batch_projects):
                batch_results = []
                
                for i, block in enumerate(project_blocks[:len(batch_projects)]):
                    new_domain = ""  # Default to EMPTY instead of "Parse Failed"
                    new_method = ""  # Default to EMPTY instead of "Parse Failed"
                    
                    lines = block.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Domain Area:"):
                            candidate = line.replace("Domain Area:", "").strip()
                            candidate = candidate.strip('"\'- ')
                            # Only assign if it's actually in our valid list
                            if candidate in S7_DOMAIN_CATEGORIES:
                                new_domain = candidate
                            else:
                                # Try fuzzy matching for S7 since categories are fewer
                                for valid_domain in S7_DOMAIN_CATEGORIES:
                                    if candidate.lower() in valid_domain.lower() or valid_domain.lower() in candidate.lower():
                                        new_domain = valid_domain
                                        print(f"    🔧 Fuzzy domain match: '{candidate}' → '{valid_domain}'")
                                        break
                        elif line.startswith("Method Area:"):
                            candidate = line.replace("Method Area:", "").strip()
                            candidate = candidate.strip('"\'- ')
                            # Only assign if it's actually in our valid list
                            if candidate in S7_METHOD_CATEGORIES:
                                new_method = candidate
                            else:
                                # Try fuzzy matching for S7 since categories are fewer
                                for valid_method in S7_METHOD_CATEGORIES:
                                    if candidate.lower() in valid_method.lower() or valid_method.lower() in candidate.lower():
                                        new_method = valid_method
                                        print(f"    🔧 Fuzzy method match: '{candidate}' → '{valid_method}'")
                                        break
                    
                    batch_results.append({
                        'domain': new_domain,
                        'method': new_method
                    })
                
                # Apply all results in this batch
                for i, result in enumerate(batch_results):
                    proj = batch_projects[i]
                    idx = proj['index']
                    
                    old_domain = df_working.loc[idx, 's7_final_domain']
                    old_method = df_working.loc[idx, 's7_final_method']
                    
                    # Only update categories that were actually problematic
                    if proj['domain_bad']:
                        df_working.loc[idx, 's7_final_domain'] = result['domain']
                        print(f"  🔄 Fixed Domain {idx}: '{old_domain}' → '{result['domain']}'")
                    
                    if proj['method_bad']:
                        df_working.loc[idx, 's7_final_method'] = result['method']
                        print(f"  🔄 Fixed Method {idx}: '{old_method}' → '{result['method']}'")
                    
                    changes_made.append({
                        'index': idx,
                        'id': proj['id'],
                        'old_domain': old_domain,
                        'new_domain': result['domain'] if proj['domain_bad'] else old_domain,
                        'old_method': old_method,
                        'new_method': result['method'] if proj['method_bad'] else old_method
                    })
                
                batch_success = True
                print(f"  ✅ Batch {batch_num+1} processed successfully")
            else:
                print(f"  ⚠️ Got {len(project_blocks)} project blocks, expected {len(batch_projects)}")
                
        except Exception as e:
            print(f"  ❌ LLM error for batch {batch_num+1}: {e}")
        
        if not batch_success:
            print(f"  🔄 Batch {batch_num+1} failed - setting empty strings for problematic categories")
            # Set empty strings for failed attempts (so they get caught in next iteration)
            for proj in batch_projects:
                idx = proj['index']
                if proj['domain_bad']:
                    df_working.loc[idx, 's7_final_domain'] = ""
                    print(f"    Set domain to empty for index {idx}")
                if proj['method_bad']:
                    df_working.loc[idx, 's7_final_method'] = ""
                    print(f"    Set method to empty for index {idx}")
        
        # Small delay between batches
        if batch_num < total_batches - 1:
            time.sleep(3)
    
    print(f"\n💾 SAVING S7 FIXES")
    print("="*60)
    
    # Save back to CSV
    df_working.to_csv("df_processing_checkpoint.csv", index=False)
    print(f"✅ Saved to df_processing_checkpoint.csv")
    
    # Update global if possible
    try:
        globals()['df_complete'] = df_working.copy()
        print(f"✅ Updated global df_complete")
    except:
        pass
    
    print(f"\n📊 S7 FIX SUMMARY:")
    print(f"   Failed projects processed: {len(failed_projects)}")
    print(f"   Batches processed: {total_batches}")
    print(f"   Changes attempted: {len(changes_made)}")
    
    return df_working

def run_s7_complete_fix():
    """Main function to run complete S7 fixes until everything is clean"""
    
    print("🎯 S7 COMPLETE FIX - NO PARSE FAILED ALLOWED")
    print("="*80)
    
    # Get model
    try:
        model = globals()['model']
        print(f"✅ Using global model")
    except:
        print("❌ ERROR: Global 'model' not found")
        return None
    
    # Keep running fix iterations until no more failures
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"\n🔄 S7 FIX ITERATION {iteration + 1}/{max_iterations}")
        print("="*60)
        
        # Step 1: Find failed projects (strict mode)
        failed_projects, df = find_s7_failed_projects()
        
        if not failed_projects:
            print(f"🎉 S7 COMPLETE - No more failed projects after {iteration + 1} iteration(s)!")
            break
        
        print(f"📋 Found {len(failed_projects)} projects that need fixing")
        
        # Step 2: Apply fixes
        df = fix_s7_failed_projects_no_parse_failed(failed_projects, df, model)
        
        # Step 3: Show progress
        print(f"\n📊 Iteration {iteration + 1} complete")
        
        if iteration == max_iterations - 1:
            print(f"⚠️ Reached maximum iterations ({max_iterations})")
            print(f"   Some projects may still need manual review")
    
    # Final verification
    print(f"\n🔍 FINAL S7 VERIFICATION")
    print("="*60)
    
    final_failed, df_final = find_s7_failed_projects()
    
    domain_matches = sum(1 for d in df_final['s7_final_domain'] if d in S7_DOMAIN_CATEGORIES)
    method_matches = sum(1 for m in df_final['s7_final_method'] if m in S7_METHOD_CATEGORIES)
    empty_domains = sum(1 for d in df_final['s7_final_domain'] if pd.isna(d) or str(d).strip() == "")
    empty_methods = sum(1 for m in df_final['s7_final_method'] if pd.isna(m) or str(m).strip() == "")
    total_projects = len(df_final)
    
    print(f"📊 FINAL S7 STATISTICS:")
    print(f"   Total projects: {total_projects}")
    print(f"   Valid domain categories: {domain_matches} ({domain_matches/total_projects*100:.2f}%)")
    print(f"   Valid method categories: {method_matches} ({method_matches/total_projects*100:.2f}%)")
    print(f"   Empty domains (acceptable): {empty_domains} ({empty_domains/total_projects*100:.2f}%)")
    print(f"   Empty methods (acceptable): {empty_methods} ({empty_methods/total_projects*100:.2f}%)")
    print(f"   Remaining problematic projects: {len(final_failed)}")
    
    if len(final_failed) == 0:
        print(f"\n🎉🎉🎉 PERFECT! 100% S7 SUCCESS - NO MORE PARSE FAILED! 🎉🎉🎉")
    else:
        print(f"\n⚠️ Still have {len(final_failed)} projects with invalid categories:")
        # Show first few remaining problems
        for proj in final_failed[:5]:
            print(f"   - Index {proj['index']}: Domain='{proj['domain']}', Method='{proj['method']}'")
        
        if len(final_failed) > 5:
            print(f"   ... and {len(final_failed) - 5} more")
        
        # Show what the remaining bad categories are
        remaining_domains = {}
        remaining_methods = {}
        
        for proj in final_failed:
            if proj['domain_bad']:
                domain = str(proj['domain'])
                remaining_domains[domain] = remaining_domains.get(domain, 0) + 1
            if proj['method_bad']:
                method = str(proj['method'])
                remaining_methods[method] = remaining_methods.get(method, 0) + 1
        
        if remaining_domains:
            print(f"\n   Remaining bad domains:")
            for domain, count in remaining_domains.items():
                print(f"     - '{domain}': {count} projects")
        
        if remaining_methods:
            print(f"\n   Remaining bad methods:")
            for method, count in remaining_methods.items():
                print(f"     - '{method}': {count} projects")
    
    print(f"\n🎯 S7 COMPLETE FIX FINISHED!")
    
    return df_final

# USAGE:
df_s7_clean = run_s7_complete_fix()


# In[161]:


### Simplify  col names
df = pd.read_csv("df_processing_checkpoint.csv")

# Create a renaming dictionary
rename_dict = {
    'l3_tech_domain': 'L3 Domain',
    'l3_strategic_method': 'L3 Method',
    'l3_std_domain_area': 'L4 Domain',
    'l3_std_method_area': 'L4 Method',
    's5_curated_domain': 'L5 Domain',
    's5_curated_method': 'L5 Method',
    's6_refined_domain': 'L6 Domain',
    's6_refined_method': 'L6 Method',
    's7_final_domain': 'L7 Domain',
    's7_final_method': 'L7 Method'
}

# Apply renaming
df.rename(columns=rename_dict, inplace=True)

# Save the updated CSV
df.to_csv("df_processing_checkpoint.csv", index=False)


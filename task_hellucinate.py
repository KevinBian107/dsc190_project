from LLM import *
from prompt import *
from env_create import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import sys
import os

def run_exp(Saving_path, pg_row_num, pg_column_num, iteration_num, query_time_limit, dialogue_history_method='_w_only_state_action_history', cen_decen_framework='HMAS-2', model_name='gpt-4'):
    Saving_path_result = Saving_path + f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/{cen_decen_framework}{dialogue_history_method}_{model_name}'
    
    # Set up directories for saving results
    os.makedirs(Saving_path_result, exist_ok=True)
    os.makedirs(Saving_path_result + f'/prompt', exist_ok=True)
    os.makedirs(Saving_path_result + f'/response', exist_ok=True)
    os.makedirs(Saving_path_result + f'/pg_state', exist_ok=True)
    os.makedirs(Saving_path_result + f'/dialogue_history', exist_ok=True)

    create_initial_pg_state_file(Saving_path, pg_row_num, pg_column_num, iteration_num)
    
    with open(Saving_path + f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'r') as file:
        pg_dict = json.load(file)

    user_prompt_list = []  # Record list of input prompts
    response_total_list = []  # Record list of responses
    pg_state_list = [pg_dict]  # Record list of pg states at varied steps
    dialogue_history_list = []
    token_num_count_list = []
    
    with open(Saving_path_result + '/pg_state' + '/pg_state' + str(1) + '.json', 'w') as f:
        json.dump(pg_dict, f)
    
    print(f'query_time_limit: {query_time_limit}')
    
    for index_query_times in range(query_time_limit):
        print(pg_dict)
        state_update_prompt = state_update_func(pg_row_num, pg_column_num, pg_dict)

        if cen_decen_framework not in ('DMAS'):
            # Each agent generates its own centralized plan
            agent_plans = {}
            for agent_row_i in range(pg_row_num):
                for agent_col_j in range(pg_column_num):
                    agent_id = f'Agent[{agent_row_i + 0.5}, {agent_col_j + 0.5}]'
                    
                    # Create prompts for each agent acting as the central planner
                    user_prompt = input_prompt_func(state_update_prompt, agent_id, pg_state_list, response_total_list, dialogue_history_list, dialogue_history_method, cen_decen_framework)
                    user_prompt_list.append(user_prompt)
                    messages = message_construct_func([user_prompt], [], '_w_all_dialogue_history')
                    
                    # Query LLM for each agent's plan
                    response, token_num_count = LLaMA_response(messages, model_name)
                    token_num_count_list.append(token_num_count)
                    print(f'Initial response from {agent_id}: ', response)
                    
                    match = re.search(r'{.*}', response, re.DOTALL)
                    if match:
                        agent_response = match.group()
                        agent_plans[agent_id] = agent_response
                    else:
                        raise ValueError(f'Response format error from {agent_id}: {response}')

            # Judge LLM to evaluate all agent plans and select the best one
            judging_prompt = create_judging_prompt(agent_plans)
            judge_response, judge_token_count = LLaMA_response([judging_prompt], model_name)
            token_num_count_list.append(judge_token_count)
            
            # Extract and check judged response
            match = re.search(r'{.*}', judge_response, re.DOTALL)
            if match:
                final_plan = match.group()
                print('Final judged plan:', final_plan)
            else:
                raise ValueError('Judge response format error:', judge_response)

            response_total_list.append(final_plan)
            dialogue_history_list.append(f'Judge: {final_plan}')
            
            # Save the judged plan
            with open(Saving_path_result + '/response' + f'/response{index_query_times + 1}.json', 'w') as f:
                json.dump(final_plan, f)

            # Update the pg_dict based on the chosen plan
            try:
                system_error_feedback, pg_dict_returned = action_from_response(pg_dict, json.loads(final_plan))
                if system_error_feedback != '':
                    print(system_error_feedback)
                pg_dict = pg_dict_returned
            except Exception as e:
                success_failure = 'Hallucination of wrong plan'
                return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result
            
            pg_state_list.append(pg_dict)
            with open(Saving_path_result + '/pg_state' + f'/pg_state{index_query_times + 2}.json', 'w') as f:
                json.dump(pg_dict, f)
            
            # Task success check
            count = sum(len(value) for value in pg_dict.values())
            if count == 0:
                break

    success_failure = 'success' if index_query_times < query_time_limit - 1 else 'failure over query time limit'
    return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result

def input_prompt_func(state_update_prompt, agent_id, pg_state_list, response_total_list, dialogue_history_list, dialogue_history_method, cen_decen_framework):
    # Construct prompt for each agent as central planner
    return f"Agent {agent_id} is imagining it is the central planner. {state_update_prompt}."

def create_judging_prompt(agent_plans):
    # Construct a prompt to let the judge LLM evaluate all plans
    return f"The following are plans proposed by different agents acting as centralized leaders: {agent_plans}. Evaluate and select the most coherent and effective plan."

def create_initial_pg_state_file(Saving_path, pg_row_num, pg_column_num, iteration_num):
    # Function to create initial pg_state file if it doesn't exist
    path = Saving_path + f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        initial_pg_state = {
            # Define the initial state structure here
            "example_key": "example_value"
        }
        with open(path, 'w') as file:
            json.dump(initial_pg_state, file)

# Example run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
Code_dir_path = os.path.join(os.getcwd(), 'runs')
os.makedirs(Code_dir_path, exist_ok=True)
Saving_path = Code_dir_path + '/Envq_BoxNet1'

model_name = "gpt-4"

print(f'-------------------Model name: {model_name}-------------------')
for pg_row_num, pg_column_num in [(2, 2), (2, 4), (4, 4), (4, 8)]:
    query_time_limit = 40 if (pg_row_num == 4 and pg_column_num == 8) else 30
    for iteration_num in range(2):
        print('-------###-------###-------###-------')
        print(f'Row num is: {pg_row_num}, Column num is: {pg_column_num}, Iteration num is: {iteration_num}\n\n')

        user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result = run_exp(
            Saving_path, pg_row_num, pg_column_num, iteration_num, 
            query_time_limit, dialogue_history_method='_w_only_state_action_history', 
            cen_decen_framework='HMAS-2', model_name=model_name
        )

        with open(Saving_path_result + '/token_num_count.txt', 'w') as f:
            for token_num in token_num_count_list:
                f.write(str(token_num) + '\n')

        with open(Saving_path_result + '/success_failure.txt', 'w') as f:
            f.write(success_failure)

        with open(Saving_path_result + '/env_action_times.txt', 'w') as f:
            f.write(f'{index_query_times + 1}')
        
        print(success_failure)
        print(f'Iteration number: {index_query_times + 1}')

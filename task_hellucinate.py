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

def create_planning_prompt(agent_id, state_update_prompt, pg_dict, other_agents):
    """
    Creates a prompt for an agent to imagine being the central planner.
    """
    prompt = f"""You are {agent_id} taking on the role of central coordinator. Analyze the environment and create a plan for all agents.

Environment State:
{state_update_prompt}

Current State Dictionary:
{json.dumps(pg_dict, indent=2)}

Other Agents: {', '.join(other_agents)}

Create a complete plan for all agents, considering:
1. Optimal coordination to minimize total moves
2. Avoiding collisions and conflicts
3. Efficient box movement paths
4. Current positions and accessibility

Format your response as a JSON dictionary mapping agent IDs to actions like:
{{
    "Agent[0.5, 0.5]": "move(box_blue, square[1.5, 1.5])",
    "Agent[1.5, 0.5]": "move(box_red, square[0.5, 1.5])"
}}

Only output the JSON plan, no explanation needed."""
    return prompt

def create_judge_prompt(agent_plans, state_update_prompt, pg_dict):
    """
    Creates a prompt for the judge to evaluate multiple agent plans.
    """
    prompt = f"""As an efficient judge, evaluate these agent-proposed plans:

Environment State:
{state_update_prompt}

Current State:
{json.dumps(pg_dict, indent=2)}

Proposed Plans:
{json.dumps(agent_plans, indent=2)}

Choose or combine the most efficient plan considering:
1. Minimal number of moves needed
2. Optimal coordination between agents
3. Avoiding potential conflicts
4. Path efficiency for each agent

Output only the final chosen plan as a JSON dictionary mapping agent IDs to actions. Use the exact same format as the input plans."""
    return prompt

def run_exp(Saving_path, pg_row_num, pg_column_num, iteration_num, query_time_limit, 
            dialogue_history_method='_w_only_state_action_history', 
            model_name='gpt-4'):

    #-----------------------------------------DEFINING-----------------------------------------#
    Saving_path_result = Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/role_playing_llm_{dialogue_history_method}_{model_name}'

    # Create directories
    os.makedirs(Saving_path_result, exist_ok=True)
    for dir_name in ['prompt', 'response', 'pg_state', 'dialogue_history', 'agent_plans']:
        os.makedirs(Saving_path_result+f'/{dir_name}', exist_ok=True)

    # Load initial state
    with open(Saving_path+f'/env_pg_state_{pg_row_num}_{pg_column_num}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'r') as file:
        pg_dict = json.load(file)

    # Initialize tracking lists
    user_prompt_list = []
    response_total_list = []
    pg_state_list = [pg_dict]
    dialogue_history_list = []
    token_num_count_list = []
    
    # Save initial state
    with open(Saving_path_result+'/pg_state/pg_state1.json', 'w') as f:
        json.dump(pg_dict, f)

    #-----------------------------------------MAIN LOOP-----------------------------------------#
    print(f'query_time_limit: {query_time_limit}')
    for index_query_times in range(query_time_limit):
        print(f'Step {index_query_times + 1}')
        print(f'Current state: {pg_dict}')

        # Get environment state description
        state_update_prompt = state_update_func(pg_row_num, pg_column_num, pg_dict)

        #-----------------------------------------AGENT ROLE-PLAYING-----------------------------------------#
        agent_plans = {}
        # Generate list of all possible agents
        all_agents = [f'Agent[{i+0.5}, {j+0.5}]' for i in range(pg_row_num) for j in range(pg_column_num)]
        
        for agent_id in all_agents:
            other_agents = [a for a in all_agents if a != agent_id]
            
            # Create planning prompt for current agent
            planning_prompt = create_planning_prompt(
                agent_id,
                state_update_prompt,
                pg_dict,
                other_agents
            )
            
            # Save agent's prompt
            with open(Saving_path_result+f'/prompt/agent_{agent_id}_prompt_{index_query_times+1}.txt', 'w') as f:
                f.write(planning_prompt)
            
            # Get agent's plan
            messages = [{"role": "user", "content": planning_prompt}]
            try:
                plan_response, token_count = LLaMA_response(messages, model_name)
                token_num_count_list.append(token_count)
                
                # Extract JSON plan
                match = re.search(r'{.*}', plan_response, re.DOTALL)
                if match:
                    plan = json.loads(match.group())
                    agent_plans[agent_id] = plan
            except Exception as e:
                print(f"Error in plan from {agent_id}: {str(e)}")
                continue

        #-----------------------------------------PLAN JUDGING-----------------------------------------#
        if agent_plans:
            judge_prompt = create_judge_prompt(agent_plans, state_update_prompt, pg_dict)
            
            # Save judge prompt
            with open(Saving_path_result+f'/prompt/judge_prompt_{index_query_times+1}.txt', 'w') as f:
                f.write(judge_prompt)
            
            messages = [{"role": "user", "content": judge_prompt}]
            judge_response, token_count = LLaMA_response(messages, model_name)
            token_num_count_list.append(token_count)
            
            match = re.search(r'{.*}', judge_response, re.DOTALL)
            if match:
                try:
                    final_plan = json.loads(match.group())
                    response = json.dumps(final_plan)
                except Exception as e:
                    print(f"Error parsing judge's decision: {str(e)}")
                    success_failure = 'Judging Error'
                    return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result
            else:
                success_failure = 'Invalid Judge Response'
                return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result
        else:
            success_failure = 'No Valid Plans'
            return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result

        #-----------------------------------------EXECUTE PLAN-----------------------------------------#
        response_total_list.append(response)
        
        # Save responses and plans
        with open(Saving_path_result+f'/response/response{index_query_times+1}.json', 'w') as f:
            json.dump(final_plan, f)
        with open(Saving_path_result+f'/agent_plans/plans{index_query_times+1}.json', 'w') as f:
            json.dump(agent_plans, f)
            
        # Execute actions
        try:
            system_feedback, new_pg_dict = action_from_response(pg_dict, final_plan)
            if system_feedback:
                print(f"System feedback: {system_feedback}")
            pg_dict = new_pg_dict
        except Exception as e:
            print(f"Execution error: {str(e)}")
            success_failure = 'Execution Error'
            return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result

        # Update and save state
        pg_state_list.append(pg_dict)
        with open(Saving_path_result+f'/pg_state/pg_state{index_query_times+2}.json', 'w') as f:
            json.dump(pg_dict, f)

        # Check if task is complete
        if sum(len(value) for value in pg_dict.values()) == 0:
            success_failure = 'success'
            break
    else:
        success_failure = 'failure over query time limit'

    return user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result

# Main execution code
if __name__ == "__main__":
    Code_dir_path = os.path.join(os.getcwd(), 'runs')
    os.makedirs(Code_dir_path, exist_ok=True)
    saving_path = Code_dir_path + '/Envq_BoxNet1'

    model_name = "qwen2.5:14b-instruct-q3_K_L"  # or your preferred model

    print(f'-------------------Model name: {model_name}-------------------')
    
    # Test configurations
    for pg_row_num, pg_column_num in [(2,2), (2,4), (4,4), (4,8)]:
        query_time_limit = 40 if (pg_row_num == 4 and pg_column_num == 8) else 30
        
        for iteration_num in range(2):
            print('-------###-------###-------###-------')
            print(f'Row num is: {pg_row_num}, Column num is: {pg_column_num}, Iteration num is: {iteration_num}\n\n')

            try:
                results = run_exp(
                    saving_path, 
                    pg_row_num, 
                    pg_column_num, 
                    iteration_num,
                    query_time_limit,
                    dialogue_history_method='_w_only_state_action_history',
                    model_name=model_name
                )
                
                user_prompt_list, response_total_list, pg_state_list, success_failure, index_query_times, token_num_count_list, Saving_path_result = results

                # Save results
                with open(Saving_path_result + '/token_num_count.txt', 'w') as f:
                    for count in token_num_count_list:
                        f.write(str(count) + '\n')

                with open(Saving_path_result + '/success_failure.txt', 'w') as f:
                    f.write(success_failure)

                with open(Saving_path_result + '/env_action_times.txt', 'w') as f:
                    f.write(f'{index_query_times+1}')

                print(f'Result: {success_failure}')
                print(f'Iteration number: {index_query_times+1}')

            except Exception as e:
                print(f"Error in experiment: {str(e)}")
                continue
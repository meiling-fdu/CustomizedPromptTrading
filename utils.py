import math
import numpy as np
import random
seed = 40
random.seed(seed)
np.random.seed(seed)


def single_quality_cal(ClipIQA, Brisque, ClipScore):

    chosen_quality = (ClipIQA - Brisque*1.0/200)/2 + ClipScore*1.0/100  # [-2,2]
    # scale to [0,1]
    chosen_quality = (chosen_quality+2)*1.0/4
    return chosen_quality


def AAA_cal(chosen_categories, a_i_list, categories_qualities):
    '''
    calculate the value of A in the paper
    chosen_categories: len() = K
    '''
    AAA_value = 0.0
    for c_idx in chosen_categories:
        cur_a_i = a_i_list[c_idx]
        cur_q_bar_i_t = categories_qualities[c_idx]
        cur_item = 1.0 / (2 * cur_a_i * cur_q_bar_i_t)
        AAA_value += cur_item
    
    return AAA_value


def BBB_cal(chosen_categories, a_i_list, b_i_list):
    '''
    calculate the value of B in the paper
    chosen_categories: len() = K
    '''
    BBB_value = 0.0
    for c_idx in chosen_categories:
        cur_a_i = a_i_list[c_idx]
        cur_b_i = b_i_list[c_idx]
        cur_item = cur_b_i / (2 * cur_a_i)
        BBB_value += cur_item
    
    return BBB_value


def Gamma_cal(AAA, gamma):
    '''
    calculate the value of Gamma in the paper
    '''
    Gamma = AAA / (2*(1+gamma*AAA))
    return Gamma


def Theta_cal(AAA, BBB, delta, gamma):
    '''
    calculate the value of Theta in the paper
    '''
    Theta = (delta*AAA - 2*gamma*AAA*BBB - BBB) / (2*(1+gamma*AAA)) + BBB
    return Theta


def Delta_cal(Gamma, Theta, eta, q_bar_t):
    '''
    calculate the value of Delta in the paper
    '''
    Delta = (2.0-Theta*q_bar_t)**2 + 8*eta*Gamma*((q_bar_t)**2)
    return Delta


def Bound_constraint(value, margin_min, margin_max):

    return max(min(value, margin_max), margin_min)
    

def OptimalFunc(chosen_categories, chosen_image_qualitys, chosen_TICorrelations, categories_qualities, HyperData, category_number, 
                gamma=0.1, delta=0.01, eta=1, omega=1, sigma_min=0.05, sigma_max=1000000000000, PT_min=0.5, PT_max=1000, PGT_min=0.5, PGT_max=1000):

    q_bar_t = sum(categories_qualities)*1.0 / len(chosen_categories) 
    # a_i, b_i  #  [0.1, 0.5] and [0.1, 1]
    a_i_list, b_i_list = HyperData['a_i'].tolist(), HyperData['b_i'].tolist()
    
    AAA = AAA_cal(chosen_categories, a_i_list, categories_qualities)
    BBB = BBB_cal(chosen_categories, a_i_list, b_i_list)
    Theta = Theta_cal(AAA, BBB, delta, gamma)
    Gamma = Gamma_cal(AAA, gamma)
    Delta = Delta_cal(Gamma, Theta, eta, q_bar_t)
    
    # Consumer
    PGT_best = (3*q_bar_t*Theta - 2.0 + np.sqrt(Delta)) / (4*q_bar_t*Gamma)
    PGT_best = Bound_constraint(PGT_best, PGT_min, PGT_max)
    
    # Platform
    PT_best = (PGT_best*AAA - (delta*AAA - 2*gamma*AAA*BBB - BBB)) / (2*AAA*(1+gamma*AAA))
    PT_best = Bound_constraint(PT_best, PT_min, PT_max)
    
    # Sellers
    sigma_best = [0 for _ in range(category_number)]
    for c_idx in range(category_number):
        if c_idx in chosen_categories:
            cur_q_bar_i_t = categories_qualities[c_idx]
            sigma_best[c_idx] = (PT_best - b_i_list[c_idx]*cur_q_bar_i_t) / (2*a_i_list[c_idx]*cur_q_bar_i_t) 
            sigma_best[c_idx] = Bound_constraint(sigma_best[c_idx], sigma_min, sigma_max)
        
    # Then WE GET: PGT_best, PT_best, sigma_best (list)

    # print("The optimal values: ", set(sigma_best), PT_best, PGT_best)
    
    '''
    Calculate profits based on the optimal strategies
    '''
    
    c_i_list = [0.01 for _ in range(category_number)]
    selection_signal = [1 if c_idx in chosen_categories else 0 for c_idx in range(category_number) ]

    category_revenue_list, category_cost_list, category_profit_list = [0 for _ in range(category_number)], [0 for _ in range(category_number)], [0 for _ in range(category_number)]
    
    # Sellers Profit calculation
    for c_idx in range(category_number):
        if c_idx in chosen_categories:
            # the revenue of category c_idx  (the first item of Equation (1))
            category_revenue_list[c_idx] = PT_best * sigma_best[c_idx] * selection_signal[c_idx]
            
            # the cost of category c_idx   (the second item of Equation (1))
            cur_q_bar_i_t = categories_qualities[c_idx]
            category_cost_list[c_idx] = ((a_i_list[c_idx]*(sigma_best[c_idx]**2)+b_i_list[c_idx]*sigma_best[c_idx])*cur_q_bar_i_t + c_i_list[c_idx] * chosen_image_qualitys[c_idx]) * selection_signal[c_idx]
            # print(category_revenue_list[c_idx], category_cost_list[c_idx])
            category_profit_list[c_idx] = category_revenue_list[c_idx] - category_cost_list[c_idx]
            
    # Platform Profit calculation
    total_entity_richness = 0.0
    for c_idx in range(category_number):
        total_entity_richness += sigma_best[c_idx] * selection_signal[c_idx]
    platform_revenue = PGT_best * total_entity_richness
    platform_cost = PT_best * total_entity_richness + gamma*(total_entity_richness**2) + delta * total_entity_richness
    platform_profit = platform_revenue - platform_cost
    
    # Consumer Profit calculation
    total_correlation_scores = sum(chosen_TICorrelations)
    consumer_revenue = eta * math.log(1+q_bar_t*total_entity_richness) + omega * total_correlation_scores
    consumer_profit = consumer_revenue - platform_revenue

    return category_profit_list, platform_profit, consumer_profit, sigma_best, PT_best, PGT_best

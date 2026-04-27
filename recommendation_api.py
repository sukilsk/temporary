from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import sqlite3
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
CORS(app) 





import pandas as pd
import sqlite3
from mlxtend.frequent_patterns import apriori, association_rules

# step 1: Find member's most frequently purchased coffee

def get_member_favorite_coffee(member_id):
    
    """Find ALL coffee beverages a member buys most often (handles ties)"""
    
    conn = sqlite3.connect('MIS_coffee.db')
    query = """
    WITH purchase_counts AS (
        SELECT 
            p.name as coffee_name,
            COUNT(*) as purchase_count
        FROM transactions t
        JOIN transaction_items ti ON t.transaction_id = ti.transaction_id
        JOIN products p ON ti.product_id = p.product_id
        WHERE t.member_id = ? AND p.category = 'coffee beverage'
        GROUP BY p.name
    ),
    max_count AS (
        SELECT MAX(purchase_count) as max_count
        FROM purchase_counts
    )
    SELECT coffee_name
    FROM purchase_counts
    WHERE purchase_count = (SELECT max_count FROM max_count)
    """
    result = pd.read_sql_query(query, conn, params=[member_id])
    conn.close()
    
    if result.empty:
        return []
    
    return result['coffee_name'].tolist()


# step 2: Get food recommendations for a specific coffee

def get_transaction_data():
    
    """Load all beverage and food transactions"""
    
    conn = sqlite3.connect('MIS_coffee.db')
    query = """
    SELECT 
        t.transaction_id,
        p.name AS product_name
    FROM transactions t
    JOIN transaction_items ti ON t.transaction_id = ti.transaction_id
    JOIN products p ON ti.product_id = p.product_id
    WHERE p.category IN ('coffee beverage', 'food')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def create_basket(df):
    
    """Convert to basket format"""
    
    basket = df.groupby(['transaction_id', 'product_name'])['product_name'].count().unstack().fillna(0)
    basket = basket.map(lambda x: True if x > 0 else False)
    return basket


def get_association_rules(basket):
    
    """Generate association rules"""
    
    frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    return rules
    

def get_popularity_for_coffee(rules, coffee_name):
    
    """
    Get popularity scores for foods paired with this specific coffee.
    Fixes the 'partial name' bug by using set-based matching.
    """
    
    if isinstance(coffee_name, list):
        coffee_name = coffee_name[0] if coffee_name else None
    
    if coffee_name is None:
        return {}
    
    scores = {}
    
    for _, row in rules.iterrows():
        ante = row['antecedents']
        cons_list = list(row['consequents'])
        
        if len(ante) == 1 and coffee_name in ante:
            
            for food_item in cons_list:
                scores[food_item] = {
                    'food': food_item,
                    'popularity': row['confidence'] * 100
                }
                
    return scores
    
    
def get_coffee_flavors(coffee_name):
    
    """Get flavor categories for a coffee beverage"""
    
    conn = sqlite3.connect('MIS_coffee.db')
    query = """
    SELECT GROUP_CONCAT(fn.category) as flavors
    FROM coffee_beverages cbv
    JOIN products p ON cbv.product_id = p.product_id
    JOIN beverage_bean_recipes bbr ON cbv.beverage_id = bbr.beverage_id
    JOIN coffee_beans cb ON bbr.coffee_bean_id = cb.coffee_bean_id
    LEFT JOIN coffee_flavors cf ON cb.coffee_bean_id = cf.coffee_bean_id
    LEFT JOIN flavor_notes fn ON cf.flavor_id = fn.flavor_id
    WHERE p.name = ? AND bbr.is_default = 1
    GROUP BY cbv.beverage_id
    """
    result = pd.read_sql_query(query, conn, params=[coffee_name])
    conn.close()
    
    if result.empty:
        return []
    
    flavors = result['flavors'].iloc[0]
    return flavors.split(',') if flavors else []


def get_coffee_purchase_count(member_id, coffee_name):
    """Get the number of times a member purchased a specific coffee"""
    
    conn = sqlite3.connect('MIS_coffee.db')
    query = """
    SELECT COUNT(*) as purchase_count
    FROM transactions t
    JOIN transaction_items ti ON t.transaction_id = ti.transaction_id
    JOIN products p ON ti.product_id = p.product_id
    WHERE t.member_id = ? AND p.name = ? AND p.category = 'coffee beverage'
    """
    result = pd.read_sql_query(query, conn, params=[member_id, coffee_name])
    conn.close()
    
    if not result.empty:
        return result.iloc[0]['purchase_count']
    return 0


def get_all_foods():
    
    """Get all food items with their attributes"""
    
    conn = sqlite3.connect('MIS_coffee.db')
    query = """
    SELECT 
        p.name as food_name,
        f.is_contain_fruit,
        f.is_contain_vegetables,
        f.is_contain_nuts,
        f.is_contain_chocolate,
        f.is_contain_cheese,
        f.is_buttery,
        f.is_savory,
        f.is_light_pastry,
        f.is_rich_pastry
    FROM food f
    JOIN products p ON f.product_id = p.product_id
    """
    foods = pd.read_sql_query(query, conn)
    conn.close()
    return foods


def calculate_flavor_score(coffee_flavors, food):
    
    """Calculate flavor match score using research-based coupling rules"""
    
    coupling_rules = {
        'fruity': ['is_contain_fruit', 'is_light_pastry', 'is_contain_cheese'],
        'floral': ['is_contain_fruit', 'is_light_pastry'],
        'sweet': ['is_buttery', 'is_contain_nuts'],
        'nutty': ['is_contain_nuts', 'is_buttery', 'is_light_pastry'],
        'roasted': ['is_contain_chocolate', 'is_savory', 'is_contain_nuts', 'is_rich_pastry'],
        'vegetative': ['is_contain_vegetables', 'is_savory'],
        'spices': ['is_contain_cheese', 'is_savory', 'is_contain_nuts'],
        'fermented': ['is_contain_cheese', 'is_savory']
    }
    
    match_score = 0
    
    # Define the weight of each match (With 35 points, 3 matches will reach ~100%)
    points_per_match = 35.0 
    
    for flavor in coffee_flavors:
        flavor = flavor.strip().lower()
        
        if flavor in coupling_rules:
            for attr in coupling_rules[flavor]:
                if food.get(attr, 0) == 1:
                    match_score += points_per_match
    
    final_score = min(100, match_score)
    
    return round(final_score, 1)


def get_flavor_scores_for_all_foods(coffee_flavors, all_foods):
    
    """Calculate flavor scores for all foods"""
    
    scores = {}
    for _, food in all_foods.iterrows():
        food_name = food['food_name']
        scores[food_name] = calculate_flavor_score(coffee_flavors, food)
    return scores


def get_popularity_recommendations(popularity_scores):
    
    """Get top foods by popularity score only"""
    
    results = []
    for food_name, data in popularity_scores.items():
        results.append({
            'food': food_name,
            'popularity': data['popularity']
        })
    
    results.sort(key=lambda x: x['popularity'], reverse=True)
    return results[:1]


def get_flavor_recommendations(flavor_scores):
    
    """Get top foods by flavor score only"""
    
    results = []
    for food_name, score in flavor_scores.items():
        results.append({
            'food': food_name,
            'flavor_score': score
        })
    
    results.sort(key=lambda x: x['flavor_score'], reverse=True)
    return results[:3]


# ============================================================
# MAIN FUNCTION: Personalized food recommendation
# ============================================================

def recommend_food_for_member(member_id):
    
    """
    Recommend food for EACH of member's favorite coffees
    Returns a dictionary with coffee names as keys
    """
    
    favorite_coffees = get_member_favorite_coffee(member_id)
    
    if not favorite_coffees:
        print(f"Member {member_id} has no coffee purchase history")
        return {}
    
    df = get_transaction_data()
    basket = create_basket(df)
    rules = get_association_rules(basket)
    all_foods = get_all_foods()
    
    results = {}
    
    for coffee in favorite_coffees:
        # Get popularity scores
        popularity_scores = get_popularity_for_coffee(rules, coffee)
        
        # Get flavor scores
        coffee_flavors = get_coffee_flavors(coffee)
        flavor_scores = get_flavor_scores_for_all_foods(coffee_flavors, all_foods)
        
        # Get recommendations
        popularity_recs = get_popularity_recommendations(popularity_scores)
        flavor_recs = get_flavor_recommendations(flavor_scores)
        
        results[coffee] = {
            'popularity': popularity_recs,
            'flavor': flavor_recs
        }
    
    return results





@app.route('/api/members', methods=['GET'])
def get_members():
    """Return list of all members for dropdown"""
    conn = sqlite3.connect('MIS_coffee.db')
    members = pd.read_sql_query("SELECT member_id, name FROM members ORDER BY member_id", conn)
    conn.close()
    return jsonify(members.to_dict(orient='records'))

@app.route('/api/recommendations/<int:member_id>', methods=['GET'])
def get_recommendations(member_id):
    """Return personalized food recommendations for a member"""
    results = recommend_food_for_member(member_id)
    
    formatted = []
    for coffee, recs in results.items():
        purchase_count = int(get_coffee_purchase_count(member_id, coffee))

        coffee_data = {
            'coffee': coffee,
            'flavor_notes': get_coffee_flavors(coffee),
            'purchase_count': purchase_count,
            'popularity_food': recs['popularity'][0]['food'] if recs['popularity'] else None,
            'popularity_percent': recs['popularity'][0]['popularity'] if recs['popularity'] else None,
            'flavor_foods': [r['food'] for r in recs['flavor']]
        }
        formatted.append(coffee_data)
    
    return jsonify(formatted)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
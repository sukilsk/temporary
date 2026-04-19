import streamlit as st
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






# ============================================================
# STREAMLIT UI WITH SIMPLE LOGIN (MEMBER ID ONLY)
# ============================================================

st.set_page_config(
    page_title="MIS Coffee Club - Food Pairing", 
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling with modern fonts
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=DM+Serif+Display:wght@400&family=DM+Sans:wght@400;500&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f0eb 0%, #e8e0d8 100%);
    }
    
    /* Apply Inter font to all body text, labels, buttons */
    html, body, .stApp, .stApp * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Apply Playfair Display to main title and prominent headers only */
    .main-title, .welcome-text {
        font-family: 'Playfair Display', serif !important;
        letter-spacing: -0.02em;
    }
    
    /* Headers inside expanders and other headers get Inter */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Change all text to dark/black */
    .stApp, .stApp * {
        color: #2c2c2c !important;
    }
    
    /* Login container */
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: transparent;
        border-radius: 20px;
        text-align: center;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        font-family: 'Playfair Display', serif !important;
        font-size: 4rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem;
        color: #2c2c2c !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 16px !important;
        font-weight: 400 !important;
        color: #b87333 !important;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6d4c41 0%, #5d4037 100%);
        color: white !important;
        border: none;
        border-radius: 30px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(109, 76, 65, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(109, 76, 65, 0.4);
        background: linear-gradient(135deg, #5d4037 0%, #4e342e 100%);
        color: white !important;
    }
    
    /* Logout button specific */
    .stButton > button:has(> :contains("Logout")) {
        background: #e0d6cc !important;
        color: #4a4a4a !important;
    }
    
     /* ============================================================
       SELECT BOX STYLING - Clean with white background
    ============================================================ */
    
    /* Remove rectangle background behind select box */
    .stSelectbox > div {
        background: transparent !important;
        box-shadow: none !important;
    }

    div[data-testid="stSelectbox"] {
        background: transparent !important;
    }

    div[data-testid="stSelectbox"] > div {
        background: transparent !important;
    }

    /* Selectbox label */
    .stSelectbox label {
        color: #2c2c2c !important;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Selectbox main container - FIXED HEIGHT */
    .stSelectbox > div > div {
        background-color: white !important;
        border-radius: 30px !important;
        border: 1px solid #e0d6cc !important;
        color: #2c2c2c !important;
        min-height: 45px !important;
        height: auto !important;
        line-height: 1.5 !important;
        padding: 0.6rem 1rem !important;
    }

    /* Fix text clipping */
    .stSelectbox > div > div > div {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }

    /* Dropdown menu options */
    .stSelectbox ul {
        background-color: white !important;
        border-radius: 16px !important;
        border: 1px solid #e0d6cc !important;
        padding: 0.5rem !important;
    }

    .stSelectbox li {
        background-color: white !important;
        color: #2c2c2c !important;
        border-radius: 30px !important;
        padding: 0.5rem 1rem !important;
        margin: 0 !important;
    }

    .stSelectbox li:hover {
        background-color: #e8e0d8 !important;
    }

    /* Selectbox container */
    div[data-baseweb="select"] {
        background-color: white !important;
        border-radius: 30px !important;
    }

    div[data-baseweb="select"] > div {
        background-color: white !important;
        border-radius: 30px !important;
        min-height: 45px !important;
        height: auto !important;
    }

    /* Fix selected value text clipping */
    div[data-baseweb="select"] span {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        line-height: 1.4 !important;
    }

    /* Dropdown popup menu */
    div[data-baseweb="popover"] {
        background-color: transparent !important;
    }

    div[data-baseweb="popover"] ul {
        background-color: white !important;
        border-radius: 16px !important;
        border: 1px solid #e0d6cc !important;
    }

    div[data-baseweb="popover"] li {
        background-color: white !important;
        color: #2c2c2c !important;
    }

    div[data-baseweb="popover"] li:hover {
        background-color: #e8e0d8 !important;
    }
    
    /* Recommendation card styling */
    .rec-card {
        background: white;
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #6d4c41;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #2c2c2c !important;
    }
    
    /* Flavor note badge */
    .flavor-badge {
        background: #5d4037 !important;  
        color: white !important;          
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #d4c5b5, transparent);
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6a6a6a !important;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0d6cc;
    }
    
    /* Coffee icon animation */
    @keyframes steam {
        0% { opacity: 0.3; transform: translateY(0px);}
        100% { opacity: 0.8; transform: translateY(-5px);}
    }
    .coffee-icon {
        animation: steam 2s ease-in-out infinite alternate;
        display: inline-block;
    }
    
    /* Welcome message */
    .welcome-text {
        text-align: center;
        color: #4a4a4a !important;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c2c2c !important;
    }
    
    /* Markdown text */
    p, li, span, div {
        color: #2c2c2c !important;
    }
    
    /* Warning, info, error messages */
    .stAlert {
        color: #2c2c2c !important;
    }
    
    /* Success message */
    .stAlert.success {
        color: #2c2c2c !important;
    }

    /* Make ALL button text white */
    .stButton > button,
    .stButton > button > *,
    .stButton > button span,
    .stButton > button p,
    .stButton > button div {
        color: white !important;
    }
    
    /* Hover state - keep text white */
    .stButton > button:hover,
    .stButton > button:hover > * {
        color: white !important;
    }

    /* Process steps styling with Tailwind-like classes */
    .process-container {
        margin: 2rem 0;
    }

    .process-title {
        text-align: center;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        color: #b87333 !important;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .process-subtitle {
        text-align: center;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 2rem;
        color: #2c2c2c !important;
    }

    /* Make boxes just fit the content */
    .step-card {
        background: white;
        border-radius: 20px;
        padding: 1.25rem;
        height: auto;
        min-height: auto;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid #e0d6cc;
    }

    .step-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(109,76,65,0.1);
    }

    /* STEP X styling */
    .step-number {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        color: #b87333 !important;
        margin-bottom: 0.5rem;
        line-height: 1;
        letter-spacing: 0.5px;
        flex-shrink: 0;
    }

    /* Step title styling */
    .step-title {
        font-family: 'DM Serif Display', serif !important;
        font-size: 18px !important;
        font-weight: 400 !important;
        color: #2c2c2c !important;
        margin: 0.25rem 0 0.5rem 0;
        line-height: 1.3;
        flex-shrink: 0;
    }

    /* Step description styling */
    .step-description {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        color: #6c757d !important;
        line-height: 1.4;
        flex-shrink: 0;
    }

    /* Center success message */
    div[data-testid="stAlert"] {
        text-align: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }



</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE FOR LOGIN
# ============================================================

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None


def get_member_name(member_id):
    """Get member name by ID"""
    conn = sqlite3.connect('MIS_coffee.db')
    query = "SELECT name FROM members WHERE member_id = ?"
    result = pd.read_sql_query(query, conn, params=[member_id])
    conn.close()
    
    if not result.empty:
        return result.iloc[0]['name']
    return "Member"


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login_page():
    """Display login page"""
    
    # Animated title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="coffee-icon" style="font-size: 3rem; text-align: center;">☕</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-title">MIS Coffee Club</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Sign in to discover your perfect pairings</div>', unsafe_allow_html=True)
    
    # Login form
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            # Get list of members for dropdown
            conn = sqlite3.connect('MIS_coffee.db')
            members = pd.read_sql_query("SELECT member_id, name FROM members ORDER BY member_id", conn)
            conn.close()
            
            # Member ID dropdown (no password!)
            selected_member_id = st.selectbox(
                "👤 Select Your Member ID",
                options=members['member_id'].tolist(),
                format_func=lambda x: f"{x}. {members[members['member_id']==x]['name'].iloc[0]}"
            )
            
            # Login button - centered using columns
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("☕ Login", type="primary", use_container_width=True):
                    # Get the member name
                    user_name = get_member_name(selected_member_id)
                    st.session_state.logged_in = True
                    st.session_state.user_id = selected_member_id
                    st.session_state.user_name = user_name
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MAIN APP (After Login)
# ============================================================

def show_main_app():
    """Display main recommendation app after login"""
    
    # Logout button in top right
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.rerun()
    
    # Welcome header without coffee emojis
    st.markdown(f'<div class="welcome-text">✨ Welcome back, {st.session_state.user_name}! ✨</div>', unsafe_allow_html=True)
    
    # Animated title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-title" style="font-size: 2rem;">Your Perfect Pairings</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Discover what foods pair perfectly with your favorite coffee</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    # Process section
    st.markdown('<div class="process-container">', unsafe_allow_html=True)
    st.markdown('<div class="process-title">THE PROCESS</div>', unsafe_allow_html=True)
    st.markdown('<div class="process-subtitle">How BrewMatch Works</div>', unsafe_allow_html=True)
    
    col_step1, col_step2, col_step3 = st.columns(3)
    
    with col_step1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 1</div>
            <div class="step-title">🔍 Detect Favorite Coffee</div>
            <div class="step-description">
            We query your full transaction history and count purchases per coffee beverage. 
            If you love two coffees equally, you get recommendations for both.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_step2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 2</div>
            <div class="step-title">📊 Market Basket Analysis</div>
            <div class="step-description">
            Using the Apriori algorithm with association rules, we identify which foods customers 
            most frequently purchase alongside your favorite coffee.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_step3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 3</div>
            <div class="step-title">🧪 Flavor Coupling Science</div>
            <div class="step-description">
            Each coffee's bean flavor notes (fruity, roasted, nutty, etc.) are matched against food 
            attributes using research-based coupling rules.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    
    # Get recommendations button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        find_button = st.button("🔍 Find My Perfect Pairings", type="primary", use_container_width=True)
    
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
    
    # Results section
    if find_button:
        with st.spinner("☕ Brewing your personalized recommendations..."):
            results = recommend_food_for_member(st.session_state.user_id)
            
            if not results:
                st.info("""
                **No coffee purchase history found.**  
                🎮 New here? Play our coffee preference game to discover your perfect coffee match!  
                Answer a few quick questions about your taste preferences, and we'll recommend the perfect coffee just for you!
                """)
            else:
                st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
                st.success("🎉 Here are your personalized pairings! 🎉")
                st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
                
                # Display results with distinct sections
                for idx, (coffee, recs) in enumerate(results.items(), 1):
                    flavors = get_coffee_flavors(coffee)
                    
                    if len(results) > 1:
                        coffee_title = f"Favorite Coffee #{idx}: {coffee}"
                    else:
                        coffee_title = f"Favorite Coffee: {coffee}"
                    
                    # Heading with coffee icon
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">☕</span>
                        <span style="font-size: 1.6rem; font-weight: 700; border-bottom: 2px solid #b87333; padding-bottom: 0.25rem;">
                            {coffee_title}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Content
                    if flavors:
                        unique_flavors = list(dict.fromkeys(flavors))
                        purchase_count = get_coffee_purchase_count(st.session_state.user_id, coffee)
                        
                        flavor_html = f'''
                        <div style="margin-bottom: 1rem;">
                            <div style="margin-bottom: 0.5rem;">
                                <span style="font-weight: 600; font-size: 1rem;"> Flavor Profile:</span>
                                <span style="margin-left: 0.5rem;">
                        '''
                        
                        for f in unique_flavors:
                            flavor_html += f'<span class="flavor-badge" style="margin-right: 0.5rem;">{f.title()}</span>'
                        
                        flavor_html += f'''
                                </span>
                            </div>
                            <div>
                                <span style="font-weight: 600; font-size: 1rem;"> Purchased:</span>
                                <span style="color: #6c757d; font-size: 1rem; margin-left: 0.5rem;">{purchase_count} time{'s' if purchase_count != 1 else ''}</span>
                            </div>
                        </div>
                        '''
                        st.markdown(flavor_html, unsafe_allow_html=True)
                    
                    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                    
                    col_rec1, col_rec2 = st.columns(2)
                    
                    with col_rec1:
                        if recs['popularity']:
                            st.markdown("#### 📊 What Others Buy")
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem;">
                                <span style="color: #6c757d; font-size: 1rem; line-height: 1.4;">
                                This is the #1 food item that customers most frequently purchase with {coffee}.
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
                            
                            for r in recs['popularity']:
                                st.markdown(f"""
                                <div class="rec-card">
                                    <span style="font-size: 1.2rem; font-weight: 600;">😋 {r['food']}</span><br>
                                    <span style="color: #8d6e63; font-size: 0.8rem;">
                                    {r['popularity']:.0f}% of customers buy this with {coffee}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)
                                            
                    with col_rec2:
                        if recs['flavor']:
                            st.markdown("#### 🍽️ Food Flavor Coupling Rules")
                            
                            if flavors:
                                unique_flavors = list(dict.fromkeys(flavors))
                                badge_html = ""
                                for f in unique_flavors:
                                    badge_html += f'<span class="flavor-badge" style="background: #e8e0d8; margin-right: 0.4rem;">{f.title()}</span>'
                                
                                st.markdown(f"""
                                <div style="margin-bottom: 1rem;">
                                    <span style="color: #6c757d; font-size: 0.95rem; line-height: 1.4;">
                                    The flavor notes in {coffee} {badge_html} are matched against food flavor attributes using food science.
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="margin-bottom: 1rem;">
                                    <span style="color: #6c757d; font-size: 0.85rem; line-height: 1.4;">
                                    The flavor notes in your coffee are matched against food flavor attributes using food science.
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            for r in recs['flavor']:
                                st.markdown(f"""
                                <div class="rec-card">
                                    😋 <strong>{r['food']}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.markdown('<div style="height: 4rem;"></div>', unsafe_allow_html=True)  # Spacing between sections
    
    # Footer
    st.markdown("""
    <div class="footer">
        ☕ Crafted with love for coffee lovers • Recommendations based on your taste profile
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN ROUTING
# ============================================================

if st.session_state.logged_in:
    show_main_app()
else:
    show_login_page()
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

st.title("🎬 Movie Revenue Predictor")
st.markdown("Enter movie details to predict its **worldwide box office revenue** and profitability.")

# ---  Load and preprocess the dataset ---
@st.cache_data
def load_data():
    url = "https://media.geeksforgeeks.org/wp-content/uploads/20240930185118/boxoffice.csv"
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    return df

@st.cache_data
def train_model(df):
    target = 'world_revenue'
    
    selected_features = ['domestic_revenue', 'opening_revenue', 'budget']
    
    df['budget_to_opening'] = df['budget'] / df['opening_revenue'].replace(0, 1)
    df['domestic_to_budget'] = df['domestic_revenue'] / df['budget'].replace(0, 1)
    df['domestic_to_world'] = df['domestic_revenue'] / df['world_revenue'].replace(0, 1)
    df['world_to_budget'] = df['world_revenue'] / df['budget'].replace(0, 1)
    
    selected_features.extend(['budget_to_opening', 'domestic_to_budget', 'domestic_to_world', 'world_to_budget'])
    
    X = df[selected_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, selected_features)
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Expanded hyperparameter tuning
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results = {
        'model': best_model,
        'features': selected_features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'feature_importances': pd.DataFrame(
            best_model.named_steps['regressor'].feature_importances_,
            index=selected_features,
            columns=['importance']
        ).sort_values('importance', ascending=False)
    }
    
    return results

df = load_data()

# Dataset summary
st.sidebar.subheader("Dataset Information")
st.sidebar.info(f"Number of movies: {len(df)}")

# Train model
with st.spinner("Training model... This may take a moment."):
    model_results = train_model(df)

# Display model metrics
st.sidebar.subheader("Model Performance")
st.sidebar.info(f"Training R²: {model_results['train_r2']:.2f}, RMSE: ${model_results['train_rmse']:,.2f}")
st.sidebar.info(f"Testing R²: {model_results['test_r2']:.2f}, RMSE: ${model_results['test_rmse']:,.2f}")

# Feature importance
st.sidebar.subheader("Feature Importance")
st.sidebar.dataframe(model_results['feature_importances'])

# ---  Data Exploration ---
st.subheader("Dataset Analysis")

# Create a sample visualization
with st.expander("Explore World Revenue vs Budget"):
    fig = px.scatter(df, x="budget", y="world_revenue", 
                    size="opening_revenue", color="domestic_revenue",
                    hover_name="title" if "title" in df.columns else None,
                    log_x=True, log_y=True,
                    title="World Revenue vs Budget (log scale)")
    
    breakeven_x = np.linspace(df["budget"].min(), df["budget"].max(), 100)
    breakeven_y = breakeven_x * 2.5
    fig.add_scatter(x=breakeven_x, y=breakeven_y, mode='lines', 
                   name='Breakeven (2.5x Budget)', line=dict(color='red', dash='dash'))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show profitability statistics
    df['profit'] = df['world_revenue'] - df['budget']
    df['roi'] = (df['profit'] / df['budget']) * 100
    
    profit_stats = {
        'Mean ROI': f"{df['roi'].mean():.1f}%",
        'Median ROI': f"{df['roi'].median():.1f}%",
        'Profitable Movies': f"{(df['profit'] > 0).mean()*100:.1f}%"
    }
    
    st.info(f"Industry Profitability: {profit_stats}")

# ---  User input form ---
st.subheader("Enter Movie Features")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        domestic_revenue = st.number_input("Domestic Revenue ($)", min_value=0.0, value=50000000.0, step=1000000.0)
        opening_revenue = st.number_input("Opening Revenue ($)", min_value=0.0, value=20000000.0, step=1000000.0)
    
    with col2:
        budget = st.number_input("Budget ($)", min_value=1000000.0, value=100000000.0, step=1000000.0)
        existing_world_revenue = st.number_input("Current World Revenue ($, 0 if predicting)", min_value=0.0, value=0.0, step=1000000.0)
    
    submitted = st.form_submit_button("Predict Revenue")
    
    if submitted:
        # If world revenue is already known, use it; otherwise predict it
        if existing_world_revenue > 0:
            predicted_revenue = existing_world_revenue
            prediction_note = "Using provided world revenue"
        else:
           
            budget_to_opening = budget / max(opening_revenue, 1)
            domestic_to_budget = domestic_revenue / max(budget, 1)
            
            
            avg_domestic_to_world = 0.4  # Placeholder - typically domestic is 40% of worldwide
            estimated_world = domestic_revenue / avg_domestic_to_world if avg_domestic_to_world > 0 else domestic_revenue * 2.5
            
            domestic_to_world = domestic_revenue / max(estimated_world, 1)
            world_to_budget = estimated_world / max(budget, 1)
            
            input_data = {
                "domestic_revenue": domestic_revenue,
                "opening_revenue": opening_revenue,
                "budget": budget,
                "budget_to_opening": budget_to_opening,
                "domestic_to_budget": domestic_to_budget,
                "domestic_to_world": domestic_to_world,
                "world_to_budget": world_to_budget
            }
            
           
            input_df = pd.DataFrame([input_data])
            
            predicted_revenue = model_results['model'].predict(input_df)[0]
            prediction_note = "AI predicted world revenue"
        
        st.success(f"💰 Predicted/Actual World Revenue: ${predicted_revenue:,.2f} ({prediction_note})")
        
        profit = predicted_revenue - budget
        roi = (profit / budget) * 100 if budget > 0 else 0
        
        profit_ratio = predicted_revenue / budget if budget > 0 else float('inf')
        

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Profit/Loss", f"${profit:,.0f}", f"{roi:.1f}%")
            
            # Movie industry typically needs 2-3x budget to be profitable due to marketing and distribution costs
            if profit_ratio >= 3:
                st.success(f"📈 Blockbuster Success (Revenue is {profit_ratio:.1f}× budget)")
            elif profit_ratio >= 2.5:
                st.success(f"📈 Very Profitable (Revenue is {profit_ratio:.1f}× budget)")
            elif profit_ratio >= 2:
                st.info(f"📈 Profitable (Revenue is {profit_ratio:.1f}× budget)")
            elif profit_ratio >= 1.5:
                st.warning(f"⚠️ Barely Breaking Even (Revenue is {profit_ratio:.1f}× budget)")
            elif profit_ratio >= 1:
                st.warning(f"📉 Likely Loss (Revenue is {profit_ratio:.1f}× budget)")
            else:
                st.error(f"📉 Major Financial Failure (Revenue is {profit_ratio:.1f}× budget)")
        
        with col2:
            if predicted_revenue > 0:
                domestic_pct = (domestic_revenue / predicted_revenue) * 100
                intl_revenue = predicted_revenue - domestic_revenue
                intl_pct = (intl_revenue / predicted_revenue) * 100
            else:
                domestic_pct = 0
                intl_revenue = 0
                intl_pct = 0
                

            revenue_data = pd.DataFrame({
                'Type': ['Domestic', 'International', 'Budget'],
                'Amount': [domestic_revenue, intl_revenue, budget]
            })
            
            fig = px.bar(revenue_data, x='Type', y='Amount', title="Revenue Breakdown")
            st.plotly_chart(fig, use_container_width=True)
        
        # Reality check warnings
        if opening_revenue < budget * 0.1 and predicted_revenue > budget * 2:
            st.warning("⚠️ This prediction may be optimistic. Movies with such low opening weekends rarely recover.")
        
        if domestic_revenue < budget * 0.3 and predicted_revenue > budget * 1.5:
            st.warning("⚠️ The domestic performance is concerning. International markets would need to perform exceptionally well.")
            
        # Distribution explanation
        st.subheader("Revenue Distribution")
        st.write(f"- 🇺🇸 Domestic: ${domestic_revenue:,.0f} ({domestic_pct:.1f}%)")
        st.write(f"- 🌎 International: ${intl_revenue:,.0f} ({intl_pct:.1f}%)")
        
        # Industry standards explanation
        st.subheader("Profitability Analysis")
        st.write("""
        **Movie Industry Standards:**
        - Marketing costs typically add 50-100% to the production budget
        - Studios receive ~50% of domestic box office and 40% of international
        - A movie generally needs to earn 2-2.5× its production budget to break even
        - Home video, streaming, and merchandise can provide additional revenue
        """)

# Add industry explanation
st.subheader("Understanding Movie Financial Success")
st.markdown("""
### Key Movie Industry Financial Metrics:
1. **Production Budget to Box Office Ratio**: 
   - ✅ 3.0+ = Highly Successful
   - ✅ 2.5+ = Profitable
   - ⚠️ 2.0-2.5 = Likely Break-Even
   - ❌ <2.0 = Potential Loss

2. **Domestic/International Split**:
   - Big-budget films typically target 35% domestic / 65% international
   - Strong international performance can save a weak domestic showing

3. **Opening Weekend Multiplier**:
   - Opening weekend × 2.5+ = Good "legs" (staying power)
   - <2.0 = Poor word-of-mouth, short theatrical life

4. **Ancillary Revenue**:
   - Streaming deals, merchandise, and home video typically add 10-50% beyond theatrical
""")

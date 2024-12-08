import tkinter as tk
from tkinter import messagebox, ttk
import mysql.connector
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import matplotlib.pyplot as plt
from fpdf import FPDF

# Database connection setup
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host='localhost',  # e.g., 'localhost'
            user='root',  # e.g., 'root'
            password='password',
            database='ecommerce_db'  # e.g., 'customer_db'
        )
        return connection
    except mysql.connector.Error as err:
        messagebox.showerror("Connection Error", f"Error connecting to database: {err}")
        return None


# Fetch customers
def fetch_customers():
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute("SELECT customer_id, location FROM customers")
        customers = cursor.fetchall()
        connection.close()
        return customers
    else:
        return []


def get_user_role(customer_id):
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute("SELECT is_admin FROM customers WHERE customer_id = %s", (customer_id,))
        result = cursor.fetchone()
        connection.close()
        return result[0] if result else False


# Fetch products
def fetch_products():
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute("SELECT product_id, product_name, price FROM products")
        products = cursor.fetchall()
        connection.close()
        return products
    else:
        return []


# Fetch previous purchases
def fetch_previous_purchases(customer_id):
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT p.product_name, p.price
            FROM purchases pur
            JOIN products p ON pur.product_id = p.product_id
            WHERE pur.customer_id = %s
        """, (customer_id,))
        purchases = cursor.fetchall()
        connection.close()
        return purchases
    else:
        return []


def submit_purchases(customer_id, cart):
    if not cart:
        messagebox.showwarning("Empty Cart", "Your cart is empty! Add items before submitting.")
        return

    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        for product_id in cart:
            # Ensure that customer_id and product_id are integers
            cursor.execute("INSERT INTO purchases (customer_id, product_id) VALUES (%s, %s)", 
                           (int(customer_id), int(product_id)))  # Convert to int here
        connection.commit()
        connection.close()
        messagebox.showinfo("Success", "Your purchases have been submitted!")
        cart.clear()
    else:
        messagebox.showerror("Error", "Unable to submit purchases.")


# Collaborative Filtering Recommendations
def collaborative_filtering_recommendations(customer_id):
    connection = connect_to_db()
    if connection is not None:
        # Fetch purchase data
        cursor = connection.cursor()
        cursor.execute("""
            SELECT customer_id, product_id
            FROM purchases
        """)
        data = cursor.fetchall()
        connection.close()

        # Create a customer-product matrix
        df = pd.DataFrame(data, columns=["customer_id", "product_id"])
        matrix = df.pivot_table(index="customer_id", columns="product_id", aggfunc="size", fill_value=0)

        # Calculate similarity
        similarity = cosine_similarity(matrix)
        similarity_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

        # Find similar customers
        similar_customers = similarity_df[customer_id].sort_values(ascending=False).index[1:]
        
        # Recommend products
        recommended_products = set()
        for similar_customer in similar_customers:
            similar_customer_purchases = df[df["customer_id"] == similar_customer]["product_id"].values
            recommended_products.update(similar_customer_purchases)
        
        return recommended_products
    return []

# Rule-Based Recommendations
def rule_based_recommendations(customer_id):
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        # Fetch customer's most purchased category
        cursor.execute("""
            SELECT p.category, COUNT(*)
            FROM purchases pur
            JOIN products p ON pur.product_id = p.product_id
            WHERE pur.customer_id = %s
            GROUP BY p.category
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """, (customer_id,))
        preferred_category = cursor.fetchone()

        if preferred_category:
            cursor.execute("""
                SELECT product_id, product_name
                FROM products
                WHERE category = %s
                ORDER BY sales DESC
                LIMIT 5
            """, (preferred_category[0],))
            recommendations = cursor.fetchall()
        else:
            cursor.execute("""
                SELECT product_id, product_name
                FROM products
                ORDER BY sales DESC
                LIMIT 5
            """)
            recommendations = cursor.fetchall()

        connection.close()
        return recommendations
    return []

def track_click(customer_id, recommended_product_id):
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO recommendation_clicks (customer_id, recommended_product_id) VALUES (%s, %s)",
            (customer_id, recommended_product_id)
        )
        connection.commit()
        connection.close()
    print(f"Click tracked: Customer {customer_id}, Product {recommended_product_id}")


def track_purchase_from_recommendation(customer_id, recommended_product_id):
    connection = connect_to_db()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO purchases_from_recommendations (customer_id, recommended_product_id) VALUES (%s, %s)",
            (customer_id, recommended_product_id)
        )
        connection.commit()
        connection.close()
    print(f"Purchase tracked: Customer {customer_id}, Product {recommended_product_id}")


# Show the Products page
def show_products_page(customer_id, root):
    root.destroy()
    products_window = tk.Tk()
    products_window.title("Products and Recommendations")

    # Cart and products
    cart = []
    products = fetch_products()
    previous_purchases = fetch_previous_purchases(customer_id)

    # Display products with checkboxes
    product_frame = tk.Frame(products_window)
    product_frame.pack()

    product_checkboxes = {}
    for product in products:
        var = tk.IntVar()
        checkbox = tk.Checkbutton(product_frame, text=f"{product[1]} - ${product[2]}", variable=var)
        checkbox.pack(anchor="w")
        product_checkboxes[product[0]] = var

    # Cart display
    cart_label = tk.Label(products_window, text="Cart Items:")
    cart_label.pack()
    cart_display = tk.Listbox(products_window)
    cart_display.pack()

    def add_to_cart():
        for product_id, var in product_checkboxes.items():
            if var.get() == 1 and product_id not in cart:
                cart.append(product_id)
                cart_display.insert(tk.END, f"Product ID: {product_id}")

    # Recommendations with checkboxes
    rec_label = tk.Label(products_window, text="Recommendations:")
    rec_label.pack()
    recommendations_display = tk.Frame(products_window)
    recommendations_display.pack()

    recommendations = collaborative_filtering_recommendations(customer_id) or rule_based_recommendations(customer_id)
    recommendation_checkboxes = {}
    for rec in recommendations:
        var = tk.IntVar()
        checkbox = tk.Checkbutton(recommendations_display, text=f"Product ID: {rec}", variable=var)
        checkbox.pack(anchor="w")
        recommendation_checkboxes[rec] = var

    def add_recommendations_to_cart():
        for product_id, var in recommendation_checkboxes.items():
            if var.get() == 1 and product_id not in cart:
                cart.append(product_id)
                cart_display.insert(tk.END, f"Product ID: {product_id}")

    def submit():
        submit_purchases(customer_id, cart)
        cart_display.delete(0, tk.END)
        cart.clear()

        # Function to show evaluation metrics when button is clicked
    def show_ctr_evaluation():
        evaluate_ctr()

      # Data Pipeline Button
    def run_data_pipeline():
        # Run the data cleaning pipeline
        run_data_cleaning()
        messagebox.showinfo("Data Pipeline", "Data cleaning pipeline has completed successfully.")

        # Analysis Buttons
    def run_customer_purchase_trends():
        analyze_customer_purchase_trends()

    def run_popular_product_analysis():
        popular_product_analysis()

        # Function to show report generation message when button is clicked
    def show_report_generation():
        generate_weekly_report()

    # Buttons
    tk.Button(products_window, text="Add to Cart", command=add_to_cart).pack()
    tk.Button(products_window, text="Add Recommendations to Cart", command=add_recommendations_to_cart).pack()
    tk.Button(products_window, text="Submit Purchases", command=submit).pack()
    tk.Button(products_window, text="Evaluation and Metrics (CTR)", command=show_ctr_evaluation).pack()  # CTR evaluation button
    tk.Button(products_window, text="Data Cleaning Pipeline", command=run_data_pipeline).pack()  # Data pipeline button
    tk.Button(products_window, text="Customer Purchase Trends", command=run_customer_purchase_trends).pack()  # Customer trends button
    tk.Button(products_window, text="Popular Product Analysis", command=run_popular_product_analysis).pack()  # Popular product button
    tk.Button(products_window, text="Generate Weekly Report", command=show_report_generation).pack()

    products_window.mainloop()

    # Recommendations
    rec_label = tk.Label(products_window, text="Recommendations:")
    rec_label.pack()
    recommendations_display = tk.Listbox(products_window)
    recommendations_display.pack()

    # Initial Recommendations
    recommendations = collaborative_filtering_recommendations(customer_id) or rule_based_recommendations(customer_id)
    for rec in recommendations:
        recommendations_display.insert(tk.END, f"Product ID: {rec}")

    products_window.mainloop()


def validate_customer_data():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM customers")
    customers = cursor.fetchall()
    customers_df = pd.DataFrame(customers, columns=['customer_id', 'age', 'location', 'purchase_preferences'])

    # Check for missing or null values
    if customers_df.isnull().values.any():
        print("Missing values found in customer data.")
        customers_df.dropna(inplace=True)  # Remove rows with missing values
        print("Removed rows with missing values in customer data.")
    else:
        print("No missing values in customer data.")

    # Check for duplicates
    duplicate_customers = customers_df[customers_df.duplicated()]
    if not duplicate_customers.empty:
        print(f"Found {len(duplicate_customers)} duplicate customer records.")
        customers_df.drop_duplicates(inplace=True)
        print("Removed duplicate customer records.")
    
    connection.close()
    return customers_df


def validate_product_data():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM products")
    products = cursor.fetchall()
    products_df = pd.DataFrame(products, columns=['product_id', 'product_name', 'category', 'price', 'stock_status'])

    # Check for missing or null values
    if products_df.isnull().values.any():
        print("Missing values found in product data.")
        products_df.dropna(inplace=True)  # Remove rows with missing values
        print("Removed rows with missing values in product data.")
    else:
        print("No missing values in product data.")

    # Check for duplicates
    duplicate_products = products_df[products_df.duplicated()]
    if not duplicate_products.empty:
        print(f"Found {len(duplicate_products)} duplicate product records.")
        products_df.drop_duplicates(inplace=True)
        print("Removed duplicate product records.")
    
    connection.close()
    return products_df


def validate_purchase_data():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM purchases")
    purchases = cursor.fetchall()
    purchases_df = pd.DataFrame(purchases, columns=['purchase_id', 'customer_id', 'product_id', 'purchase_date'])

    # Remove rows with missing customer_id or product_id
    purchases_df.dropna(subset=['customer_id', 'product_id'], inplace=True)

    # Remove duplicates in purchase records
    purchases_df.drop_duplicates(inplace=True)
    
    print(f"Cleaned purchase history: {len(purchases_df)} records remain.")
    connection.close()
    return purchases_df

def standardize_product_categories():
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT category FROM products")
    categories = cursor.fetchall()

    # Standardizing product categories by converting to lowercase and removing extra spaces    
    standardized_categories = {category[0].lower().strip() for category in categories}
    print("Standardized product categories:", standardized_categories)

    # Normalize categories in the products table
    cursor.execute("SELECT product_id, product_name, category FROM products")
    products = cursor.fetchall()

    for product_id, product_name, category in products:
        standardized_category = category.lower().strip()
        cursor.execute("""
            UPDATE products 
            SET category = %s 
            WHERE product_id = %s
        """, (standardized_category, product_id))
        connection.commit()

    print("Standardized product categories in the database.")
    connection.close()


def normalize_customer_ids_and_timestamps():
    connection = connect_to_db()
    cursor = connection.cursor()
    # Normalize customer IDs (ensure they are consistent)
    cursor.execute("SELECT DISTINCT customer_id FROM purchases")
    customer_ids = cursor.fetchall()

    for customer_id, in customer_ids:
        cursor.execute("""
            UPDATE purchases 
            SET customer_id = %s 
            WHERE customer_id IS NULL
        """, (customer_id,))
        connection.commit()

    # Normalize timestamps to a standard format (if needed)
    cursor.execute("""
        UPDATE purchases
        SET purchase_date = STR_TO_DATE(purchase_date, '%Y-%m-%d %H:%i:%s')
    """)
    connection.commit()

    print("Normalized customer IDs and timestamps.")
    connection.close()


# Evaluation function to calculate and visualize CTR
def evaluate_ctr():
    connection = connect_to_db()  # Establish connection
    if connection is None:
        return  # Exit if connection fails

    cursor = connection.cursor()

    # Fetch click-through data
    cursor.execute("""
        SELECT customer_id, COUNT(click_id) AS num_clicks
        FROM recommendation_clicks
        GROUP BY customer_id
    """)
    click_data = cursor.fetchall()

    # Fetch total recommendations
    cursor.execute("""
        SELECT customer_id, COUNT(*) AS total_recommendations
        FROM recommendations
        GROUP BY customer_id
    """)
    recommendation_data = cursor.fetchall()

    # Convert data to DataFrames for easy processing
    click_df = pd.DataFrame(click_data, columns=['customer_id', 'num_clicks'])
    recommendation_df = pd.DataFrame(recommendation_data, columns=['customer_id', 'total_recommendations'])

    # Merge the data on customer_id
    metrics_df = pd.merge(click_df, recommendation_df, on='customer_id', how='outer')

    # Handle potential division by zero and calculate CTR
    metrics_df['ctr'] = metrics_df.apply(lambda row: row['num_clicks'] / row['total_recommendations'] if row['total_recommendations'] > 0 else 0, axis=1)

    # Show CTR for each customer
    print("Evaluation Real-Time Metrics:")
    print(metrics_df[['customer_id', 'ctr']])

    # Visualization of CTR distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics_df['ctr'], bins=20, edgecolor='black')
    plt.title('Distribution of Click-Through Rate (CTR)')
    plt.xlabel('Click-Through Rate (CTR)')
    plt.ylabel('Number of Customers')
    plt.grid(True)
    plt.show()

    connection.close()  # Close connection after query is complete
    

# Running the full data cleaning pipeline
def run_data_cleaning():
    print("Starting data cleaning pipeline...")

    validate_customer_data()
    validate_product_data()
    purchases_df = validate_purchase_data()
    standardize_product_categories()
    normalize_customer_ids_and_timestamps()

    print("Data cleaning pipeline complete.")


def analyze_customer_purchase_trends():
    connection = connect_to_db()  # Establish connection
    if connection is None:
        return  # Exit if connection fails
    
    cursor = connection.cursor()
    
    # Fetch purchase data along with customer segment (location and age)
    query = """
        SELECT p.customer_id, c.location, c.age, p.product_id, p.purchase_date
        FROM purchases p
        JOIN customers c ON p.customer_id = c.customer_id
    """
    cursor.execute(query)
    purchase_data = cursor.fetchall()

    # Create a DataFrame for easier analysis
    purchase_df = pd.DataFrame(purchase_data, columns=['customer_id', 'location', 'age', 'product_id', 'purchase_date'])

    # Aggregate purchase data by location and age group
    location_trends = purchase_df.groupby('location')['product_id'].count().reset_index(name='total_purchases')
    age_trends = purchase_df.groupby('age')['product_id'].count().reset_index(name='total_purchases')

    # Identify top-selling products (products with the most purchases)
    product_trends = purchase_df.groupby('product_id')['product_id'].count().reset_index(name='purchase_count')
    top_selling_products = product_trends.sort_values(by='purchase_count', ascending=False).head(5)

    # Display the trends
    print("\nCustomer Purchase Trends by Location:")
    print(location_trends)
    print("\nCustomer Purchase Trends by Age Group:")
    print(age_trends)
    print("\nTop-Selling Products:")
    print(top_selling_products)

    # Plot purchase trends
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(location_trends['location'], location_trends['total_purchases'])
    plt.title("Purchase Trends by Location")
    plt.xlabel("Location")
    plt.ylabel("Total Purchases")

    plt.subplot(1, 2, 2)
    plt.bar(age_trends['age'], age_trends['total_purchases'])
    plt.title("Purchase Trends by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Total Purchases")

    plt.tight_layout()
    plt.show()

    connection.close()  # Close connection after query is complete


def popular_product_analysis():
    connection = connect_to_db()  # Establish connection
    if connection is None:
        return  # Exit if connection fails

    cursor = connection.cursor()

    # Fetch product and recommendation click data
    query = """
        SELECT p.product_id, p.product_name, COUNT(pc.click_id) AS num_clicks, COUNT(pr.purchase_id) AS num_purchases
        FROM products p
        LEFT JOIN recommendation_clicks pc ON p.product_id = pc.recommended_product_id
        LEFT JOIN purchases pr ON p.product_id = pr.product_id
        GROUP BY p.product_id, p.product_name
    """
    cursor.execute(query)
    product_data = cursor.fetchall()

    # Create DataFrame for product analysis
    product_df = pd.DataFrame(product_data, columns=['product_id', 'product_name', 'num_clicks', 'num_purchases'])

    # Identify the most frequently purchased products
    popular_products = product_df.sort_values(by='num_purchases', ascending=False).head(5)

    # Identify the most frequently clicked products
    clicked_products = product_df.sort_values(by='num_clicks', ascending=False).head(5)

    # Analyze recommendation success (purchase rate from click-through)
    product_df['purchase_rate_from_click'] = product_df['num_purchases'] / product_df['num_clicks']

    # Display the results
    print("\nMost Frequently Purchased Products:")
    print(popular_products)
    print("\nMost Frequently Clicked Products:")
    print(clicked_products)
    print("\nProduct Purchase Rate from Click-Through:")
    print(product_df[['product_name', 'purchase_rate_from_click']])

    # Plot product analysis
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(popular_products['product_name'], popular_products['num_purchases'])
    plt.title("Most Frequently Purchased Products")
    plt.xlabel("Product Name")
    plt.ylabel("Number of Purchases")

    plt.subplot(1, 2, 2)
    plt.bar(clicked_products['product_name'], clicked_products['num_clicks'])
    plt.title("Most Frequently Clicked Products")
    plt.xlabel("Product Name")
    plt.ylabel("Number of Clicks")

    plt.tight_layout()
    plt.show()

    connection.close()  # Close connection after query is complete


# Function to fetch customer purchase trends
def fetch_customer_purchase_trends():
    connection = connect_to_db()
    if connection is None:
        return pd.DataFrame()

    query = """
    SELECT p.customer_id, c.location, c.age, p.product_id, p.purchase_date
    FROM purchases p
    JOIN customers c ON p.customer_id = c.customer_id
    """
    df = pd.read_sql(query, connection)
    connection.close()
    return df


# Function to fetch popular product data
def fetch_popular_product_data():
    connection = connect_to_db()
    if connection is None:
        return pd.DataFrame()

    query = """
    SELECT p.product_id, p.product_name, COUNT(pr.purchase_id) AS num_purchases
    FROM products p
    LEFT JOIN purchases pr ON p.product_id = pr.product_id
    GROUP BY p.product_id, p.product_name
    """
    df = pd.read_sql(query, connection)
    connection.close()
    return df


# Display Customer Purchase Trends with interactive filtering
def display_purchase_trends(df):
    st.subheader("Customer Purchase Trends")
    st.write("Analyze customer purchase patterns based on location and age.")
    
    # Filtering options
    locations = st.multiselect("Select Location(s):", df['location'].unique())
    ages = st.multiselect("Select Age Group(s):", df['age'].unique())
    
    # Filter data based on user selection
    filtered_df = df
    if locations:
        filtered_df = filtered_df[filtered_df['location'].isin(locations)]
    if ages:
        filtered_df = filtered_df[filtered_df['age'].isin(ages)]

    # Plot purchase trends by location and age
    location_trends = filtered_df.groupby('location')['product_id'].count().reset_index(name='total_purchases')
    age_trends = filtered_df.groupby('age')['product_id'].count().reset_index(name='total_purchases')
    
    st.write("Purchase Trends by Location:")
    fig1 = plt.figure(figsize=(10, 6))
    plt.bar(location_trends['location'], location_trends['total_purchases'])
    plt.xlabel("Location")
    plt.ylabel("Total Purchases")
    plt.title("Purchase Trends by Location")
    st.pyplot(fig1)

    st.write("Purchase Trends by Age Group:")
    fig2 = plt.figure(figsize=(10, 6))
    plt.bar(age_trends['age'], age_trends['total_purchases'])
    plt.xlabel("Age Group")
    plt.ylabel("Total Purchases")
    plt.title("Purchase Trends by Age Group")
    st.pyplot(fig2)


# Display Popular Products with interactive filtering
def display_popular_products(df):
    st.subheader("Popular Products")
    st.write("View the most frequently purchased products.")

    # Filter for top products
    top_products = df.sort_values(by='num_purchases', ascending=False).head(10)

    fig3 = px.bar(top_products, x='product_name', y='num_purchases', title="Top 10 Most Purchased Products")
    st.plotly_chart(fig3)


# Streamlit Dashboard Layout
def run_dashboard():
    st.title("E-Commerce Analytics Dashboard")

    # Fetch the data
    purchase_df = fetch_customer_purchase_trends()
    product_df = fetch_popular_product_data()

    # Display the purchase trends
    display_purchase_trends(purchase_df)

    # Display the popular products
    display_popular_products(product_df)

    # Refresh the data every 10 seconds for real-time updates
    st.experimental_rerun()


# Function to connect to the MySQL database
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='password',
            database='ecommerce_db'
        )
        return connection
    except mysql.connector.Error as err:
        messagebox.showerror("Connection Error", f"Error connecting to database: {err}")
        return None


# Function to fetch recommendation data and CTR
def fetch_recommendation_data():
    connection = connect_to_db()
    cursor = connection.cursor()

    # Fetch click-through data (CTR) using the correct column name 'click_timestamp'
    cursor.execute("""
        SELECT customer_id, COUNT(click_id) AS num_clicks
        FROM recommendation_clicks
        WHERE click_timestamp >= CURDATE() - INTERVAL 7 DAY  # Updated to use click_timestamp
        GROUP BY customer_id
    """)
    click_data = cursor.fetchall()

    # Fetch total recommendations without filtering by recommendation_date
    cursor.execute("""
        SELECT customer_id, COUNT(*) AS total_recommendations
        FROM recommendations
        GROUP BY customer_id  # Removed the WHERE clause for recommendation_date
    """)
    recommendation_data = cursor.fetchall()

    # Convert data to DataFrames
    click_df = pd.DataFrame(click_data, columns=['customer_id', 'num_clicks'])
    recommendation_df = pd.DataFrame(recommendation_data, columns=['customer_id', 'total_recommendations'])

    # Merge the data on customer_id
    metrics_df = pd.merge(click_df, recommendation_df, on='customer_id', how='outer')

    # Calculate CTR
    metrics_df['ctr'] = metrics_df.apply(lambda row: row['num_clicks'] / row['total_recommendations'] if row['total_recommendations'] > 0 else 0, axis=1)

    return metrics_df


# Function to export data to CSV
def export_to_csv(dataframe, filename="recommendation_report.csv"):
    dataframe.to_csv(filename, index=False)
    print(f"Report saved to {filename}")


# Function to export data to PDF
def export_to_pdf(dataframe, filename="recommendation_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.cell(200, 10, txt="Weekly Recommendation Performance Report", ln=True, align='C')
    
    # Add column headers
    pdf.ln(10)  # Line break
    pdf.cell(40, 10, 'Customer ID', border=1)
    pdf.cell(40, 10, 'Number of Clicks', border=1)
    pdf.cell(40, 10, 'Total Recommendations', border=1)
    pdf.cell(40, 10, 'Click-Through Rate (CTR)', border=1)
    pdf.ln(10)

    # Add data rows
    for index, row in dataframe.iterrows():
        pdf.cell(40, 10, str(row['customer_id']), border=1)
        pdf.cell(40, 10, str(row['num_clicks']), border=1)
        pdf.cell(40, 10, str(row['total_recommendations']), border=1)
        pdf.cell(40, 10, str(row['ctr']), border=1)
        pdf.ln(10)
    
    pdf.output(filename)
    print(f"Report saved to {filename}")


# Function to generate the weekly report and export it
def generate_weekly_report():
    print("Generating weekly report...")

    # Fetch recommendation data
    metrics_df = fetch_recommendation_data()

    if metrics_df is not None:
        # Export report to CSV and PDF
        export_to_csv(metrics_df)
        export_to_pdf(metrics_df)

        print("Weekly report generation complete.")
    else:
        print("Failed to fetch data for report generation.")


# Create Login UI
def create_login_ui():
    root = tk.Tk()
    root.title("Customer Login")

    customers = fetch_customers()
    customer_dict = {f"ID: {customer[0]} (Location: {customer[1]})": customer[0] for customer in customers}

    selected_customer = tk.StringVar()
    selected_customer.set("Select a customer")
    tk.OptionMenu(root, selected_customer, *customer_dict.keys()).pack()

    def proceed():
        customer_label = selected_customer.get()
        if customer_label != "Select a customer":
            customer_id = customer_dict[customer_label]
            show_products_page(customer_id, root)

    tk.Button(root, text="Proceed", command=proceed).pack()
    root.mainloop()

if __name__ == "__main__":
    create_login_ui()

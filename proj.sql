CREATE DATABASE ecommerce_db;
USE ecommerce_db;
CREATE TABLE customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL,
    location VARCHAR(100),
    purchase_preferences TEXT
);
CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(10, 2),
    stock_status ENUM('in_stock', 'out_of_stock') DEFAULT 'in_stock'
);
CREATE TABLE purchases (
    purchase_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    product_id INT NOT NULL,
    purchase_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE TABLE recommendations (
    recommendation_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    recommended_product_ids TEXT, -- Comma-separated product IDs
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
CREATE INDEX idx_customer_id ON purchases(customer_id);
CREATE INDEX idx_product_id ON purchases(product_id);

SHOW DATABASES;
USE ecommerce_db;
SHOW TABLES;

INSERT INTO customers (age, location, purchase_preferences)
VALUES (30, 'New York', 'electronics, books'), 
       (25, 'Los Angeles', 'fashion, gadgets');
INSERT INTO products (product_name, category, price, stock_status)
VALUES ('Smartphone', 'electronics', 699.99, 'in_stock'),
       ('T-Shirt', 'fashion', 19.99, 'in_stock');
INSERT INTO purchases (customer_id, product_id)
VALUES (1, 1), 
       (2, 2);
INSERT INTO recommendations (customer_id, recommended_product_ids)
VALUES (1, '2,3'),
       (2, '1,4');
select * from customers;
SELECT * FROM products;
USE ecommerce_db;
INSERT INTO customers (age, location, purchase_preferences)
VALUES 
    (40, 'Chicago', 'sports, books'),
    (35, 'Houston', 'electronics, fashion'),
    (28, 'Miami', 'beauty, fashion');
INSERT INTO products (product_name, category, price, stock_status)
VALUES 
    ('Laptop', 'electronics', 999.99, 'in_stock'),
    ('Running Shoes', 'sports', 49.99, 'in_stock'),
    ('Lipstick', 'beauty', 9.99, 'in_stock'),
    ('Novel', 'books', 14.99, 'in_stock');
INSERT INTO purchases (customer_id, product_id)
VALUES 
    (1, 3); 
select * from customers;
SELECT * FROM products;
SELECT * FROM recommendations;

SELECT * FROM purchases WHERE customer_id = 1 ORDER BY purchase_date DESC LIMIT 5;

SELECT * FROM products;

-- Insert a new product first
INSERT INTO products (product_name, category, price, stock_status)
VALUES ('Novel', 'books', 14.99, 'in_stock');  -- This product_id should now be 3

DELIMITER $$

CREATE TRIGGER update_recommendations_after_purchase
AFTER INSERT ON purchases
FOR EACH ROW
BEGIN
    DECLARE recommended_products TEXT;
    
    -- Call a function to generate recommendations (you'll need to create it in Python or use your existing functions)
    -- For now, we'll just assume you have a stored procedure to get the recommendations based on the latest purchase
    
    -- Example logic: generate recommendations based on the most recent purchase (this can be more complex)
    SET recommended_products = (SELECT GROUP_CONCAT(product_id) 
                                FROM products 
                                WHERE category IN (SELECT category FROM products WHERE product_id = NEW.product_id)
                                AND product_id != NEW.product_id
                                LIMIT 5);
    
    -- Insert the recommendations into the recommendations table for the customer
    INSERT INTO recommendations (customer_id, recommended_product_ids)
    VALUES (NEW.customer_id, recommended_products);
    
END $$

DELIMITER ;

CREATE TABLE recommendation_clicks (
    click_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    recommended_product_id INT,
    click_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (recommended_product_id) REFERENCES products(product_id)
);

CREATE TABLE purchases_from_recommendations (
    purchase_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    recommended_product_id INT,
    purchase_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (recommended_product_id) REFERENCES products(product_id)
);

-- Example: Simulate a customer click on a recommended product
INSERT INTO recommendation_clicks (customer_id, recommended_product_id)
VALUES (1, 3), (2, 1), (1, 4);

DESCRIBE recommendations;
DESCRIBE recommendation_clicks;




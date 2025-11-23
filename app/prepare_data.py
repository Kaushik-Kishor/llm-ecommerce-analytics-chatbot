import os
import pandas as pd

DATA_RAW = "../data_raw"
DATA_PROCESSED = "../data_processed"

os.makedirs(DATA_PROCESSED, exist_ok=True)

def load_olist_data(base_path=DATA_RAW):
    customers = pd.read_csv(os.path.join(base_path, "olist_customers_dataset.csv"))
    orders = pd.read_csv(os.path.join(base_path, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(base_path, "olist_order_items_dataset.csv"))
    payments = pd.read_csv(os.path.join(base_path, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(base_path, "olist_order_reviews_dataset.csv"))
    products = pd.read_csv(os.path.join(base_path, "olist_products_dataset.csv"))
    sellers = pd.read_csv(os.path.join(base_path, "olist_sellers_dataset.csv"))
    geolocation = pd.read_csv(os.path.join(base_path, "olist_geolocation_dataset.csv"))
    categories = pd.read_csv(os.path.join(base_path, "product_category_name_translation.csv"))
    
    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "geolocation": geolocation,
        "categories": categories,
    }

def build_core_orders_table(d):
    """
    Build a single denormalized table for analytics + LLM:
    each row ~ an order item with basic customer, product, payment, review info.
    """
    orders = d["orders"]
    customers = d["customers"]
    order_items = d["order_items"]
    payments = d["payments"]
    reviews = d["reviews"]
    products = d["products"]
    sellers = d["sellers"]
    categories = d["categories"]

    # merge order + customer
    df = orders.merge(
        customers,
        on="customer_id",
        how="left",
        suffixes=("", "_customer")
    )

    # merge order items (one row per product in an order)
    df = df.merge(
        order_items,
        on="order_id",
        how="left"
    )

    # merge product info
    df = df.merge(
        products,
        on="product_id",
        how="left"
    )

    # merge category translation
    df = df.merge(
        categories,
        on="product_category_name",
        how="left"
    )

    # merge seller info
    df = df.merge(
        sellers,
        on="seller_id",
        how="left",
        suffixes=("", "_seller")
    )

    # merge payments (aggregate by order_id first – there can be multiple payments)
    pay_agg = payments.groupby("order_id").agg(
        payment_sequential_max=("payment_sequential", "max"),
        payment_installments_max=("payment_installments", "max"),
        payment_value_sum=("payment_value", "sum"),
        main_payment_type=("payment_type", lambda x: x.iloc[0])
    ).reset_index()

    df = df.merge(
        pay_agg,
        on="order_id",
        how="left"
    )

    # merge reviews (one per order_id, take latest review)
    rev_agg = reviews.sort_values("review_creation_date").groupby("order_id").tail(1)
    rev_agg = rev_agg[[
        "order_id",
        "review_score",
        "review_comment_title",
        "review_comment_message"
    ]]

    df = df.merge(
        rev_agg,
        on="order_id",
        how="left"
    )

    # create some helper columns
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["order_estimated_delivery_date"] = pd.to_datetime(df["order_estimated_delivery_date"])

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["is_late"] = (df["delivery_days"] > df["estimated_delivery_days"]).astype("int")

    # nice display name for category
    df["product_category_clean"] = df["product_category_name_english"].fillna(
        df["product_category_name"]
    )

    return df

def main():
    d = load_olist_data()
    core = build_core_orders_table(d)
    print("Rows:", len(core), "Columns:", len(core.columns))

    # Save for BI tools (Parquet is efficient + supported by Power BI/Desktop)
    core.to_parquet(os.path.join(DATA_PROCESSED, "olist_core_orders.parquet"), index=False)
    # also CSV version if you want for Tableau
    core.to_csv(os.path.join(DATA_PROCESSED, "olist_core_orders.csv"), index=False)

    print("Saved to data_processed/")

if __name__ == "__main__":
    main()

def recommend_for_customer(transactions, customer_id, segmented, limit=5):
    customer_id = str(customer_id)
    own = transactions[transactions["CustomerID"].astype(str) == customer_id]
    if own.empty:
        return []
    already_bought = set(own["Description"].astype(str))
    # Recommend frequently purchased items from the customer's own history first.
    own_top = (own.groupby("Description")["Quantity"].sum().sort_values(ascending=False).index.tolist())
    picks = [p for p in own_top if p not in ("Unknown item",)][:limit]
    if len(picks) >= limit:
        return picks
    row = segmented[segmented["CustomerID"].astype(str) == customer_id]
    if row.empty:
        return picks
    cluster = row.iloc[0]["Cluster"]
    group_ids = set(segmented.loc[segmented["Cluster"] == cluster, "CustomerID"].astype(str))
    group_data = transactions[transactions["CustomerID"].astype(str).isin(group_ids)]
    popular = (group_data.groupby("Description")["Quantity"].sum().sort_values(ascending=False).index.tolist())
    for item in popular:
        if item not in already_bought and item not in picks and item != "Unknown item":
            picks.append(item)
        if len(picks) >= limit:
            break
    return picks

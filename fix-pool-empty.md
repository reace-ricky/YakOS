## Fix: guard pool.empty in edge_tab.py

In `app/edge_tab.py`, find this line:

```python
player_names = sorted(pool["player_name"].dropna().unique().tolist()) if not pool.empty else []
```

Replace it with:

```python
player_names = sorted(pool["player_name"].dropna().unique().tolist()) if isinstance(pool, pd.DataFrame) and not pool.empty else []
```

Then commit:
```
git add app/edge_tab.py
git commit -m "fix: guard pool.empty with isinstance check"
git push origin main
```

Acceptance test: Lab tab -> Run Edge Analysis -> no red dict error box.

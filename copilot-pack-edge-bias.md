## Copilot Pack: Edge bias + `.empty` bug

### Target file: `app/edge_tab.py`

### Fix 1: `load_bias` import scope

1. Keep this at the top of the file (near other imports):
```python
from yak_core.bias import load_bias
```

2. At the very start of `_render_the_board` function body, add:
```python
bias = load_bias()
manual_fades = [n for n, v in bias.items() if v.get("max_exposure", 1.0) == 0.0]
```

3. Find every inner import inside `_render_the_board` that says:
```python
from yak_core.bias import load_bias, save_bias
```
Change each one to:
```python
from yak_core.bias import save_bias
```
The calls to `load_bias()` in those blocks are fine — they now use the top-level import.

### Fix 2: Prevent `'dict' object has no attribute 'empty'`

In all Python files, find every pattern like:
```python
if not something.empty:
```
or
```python
if something and not something.empty:
```
Replace with:
```python
if isinstance(something, (pd.DataFrame, pd.Series)) and not something.empty:
```

### Acceptance tests

1. Edge tab loads with no errors.
2. Players with `"max_exposure": 0.0` in `ricky_bias.json` appear under THE FADE.
3. Lab tab → Run Edge Analysis → no red `'dict' object has no attribute 'empty'` box.

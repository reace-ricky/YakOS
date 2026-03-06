import os
filepath = "pages/1_the_lab.py"
with open(filepath, "r") as f:
    content = f.read()

old1 = '''            pool_count = len(hub_pool)
            pmin, pmax = get_pool_size_range(contest_type_label)
            if pmin <= pool_count <= pmax:
                st.success(f"\\u2705 {pool_count} players \\u2014 in range for {contest_type_label} (target {pmin}\\u2013{pmax})")
            elif pool_count < pmin:
                st.warning(f"\\u26a0\\ufe0f {pool_count} players \\u2014 below target (need {pmin}\\u2013{pmax})")
            else:
                st.warning(f"\\u26a0\\ufe0f {pool_count} players \\u2014 above target (target {pmin}\\u2013{pmax})")'''
new1 = '            st.caption(f"{len(hub_pool)} players loaded.")'
if old1 in content:
    content = content.replace(old1, new1)
    print("Fix 1: Banner removed")
else:
    print("Fix 1: Not found - trying alternate match")
    content = content.replace("above target (target {pmin}", "MARKER_FOUND")
    if "MARKER_FOUND" in content:
        print("  Found the text but quotes differ - doing line-based fix")
        content = content.replace("MARKER_FOUND", "above target (target {pmin}")
        lines = content.split("\n")
        new_lines = []
        skip_until_blank = False
        for i, line in enumerate(lines):
            if "pool_count = len(hub_pool)" in line:
                new_lines.append('            st.caption(f"{len(hub_pool)} players loaded.")')
                skip_until_blank = True
                continue
            if skip_until_blank:
                if "above target" in line:
                    skip_until_blank = False
                continue
            new_lines.append(line)
        content = "\n".join(new_lines)
        print("  Line-based fix applied")
    else:
        print("  Banner already removed")

old2 = "dummy_lineups = _make_dummy_lineups_df(pool)"
if old2 in content:
    lines = content.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        if "dummy_lineups = _make_dummy_lineups_df" in lines[i]:
            indent = "                    "
            new_lines.append(indent + "# Build real optimized lineups instead of dummy placeholders")
            new_lines.append(indent + "_PIPELINE_TO_OPTIMIZER = {\"GPP_MAIN\": \"GPP_150\", \"GPP_EARLY\": \"GPP_20\", \"GPP_LATE\": \"GPP_20\", \"CASH\": \"CASH\"}")
            new_lines.append(indent + "optimizer_contest = _PIPELINE_TO_OPTIMIZER.get(pipeline_contest, \"GPP_20\")")
            new_lines.append(indent + "real_lineups = build_ricky_lineups(edge_df=compute_edge_metrics(pool, calibration_state=slate.calibration_state, variance=sim.variance), contest_type=optimizer_contest, calibration_state=slate.calibration_state, salary_cap=SALARY_CAP)")
            new_lines.append(indent + "if not real_lineups.empty:")
            i += 1
            while i < len(lines) and "if not dummy_lineups.empty" in lines[i]:
                i += 1
            continue
        line = lines[i].replace("dummy_lineups", "real_lineups").replace("lineups_df=real_lineups", "lineups_df=real_lineups")
        new_lines.append(line)
        i += 1
    content = "\n".join(new_lines)
    print("Fix 2: Dummy lineups replaced with optimizer")
else:
    print("Fix 2: Already applied")

with open(filepath, "w") as f:
    f.write(content)
print("Done")

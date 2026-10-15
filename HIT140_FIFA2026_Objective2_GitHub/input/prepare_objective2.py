"""Clean FBref World Cup 2026 scores and create Objective 2 match/team-match bases.

Usage: python prepare_objective2.py
This script does NOT manufacture eight pre-match predictors. It only prepares
observed responses and traceable match identifiers; additional data is required.
"""
from pathlib import Path
from collections import Counter
import csv
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'worldcup2026_matches_raw.csv'
MATCH_OUT = HERE / 'objective2_match_base_104.csv'
TEAM_OUT = HERE / 'objective2_team_match_base_208.csv'

# FBref prefixes/suffixes national team names with a country code.
# Home cell: 'Mexico mx'; away cell: 'za South Africa'.
def team_name(cell, side):
    cell = cell.strip()
    pattern = r'^(.*?)\s+([a-z]{2,3})$' if side == 'home' else r'^([a-z]{2,3})\s+(.*?)$'
    match = re.fullmatch(pattern, cell)
    if not match:
        raise ValueError(f'Unrecognized {side} team format: {cell!r}')
    return match.group(1) if side == 'home' else match.group(2)

# A penalty-shootout score such as '(3) 1–1 (4)' contains TWO different scores.
# The inner score (1–1) is the goals scored including any extra time; shootout
# kicks are not goals for the regression response.
SCORE_PATTERN = re.compile(r'^\s*(?:\((\d+)\)\s*)?(\d+)\s*[–-]\s*(\d+)(?:\s*\((\d+)\))?\s*$')

def read_source():
    with SOURCE.open('r', encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.DictReader(fh))
    matches = []
    for row in rows:
        if not any(value.strip() for value in row.values() if value is not None):
            continue
        if not row['Date'] or not row['Home'] or not row['Away']:
            raise ValueError(f'Incomplete nonempty row: {row}')
        score = SCORE_PATTERN.fullmatch(row['Score'])
        if not score:
            raise ValueError(f'Unparseable score: {row["Score"]!r}')
        pen_a, goals_a, goals_b, pen_b = score.groups()
        if (pen_a is None) != (pen_b is None):
            raise ValueError(f'Partial penalty-shootout notation: {row["Score"]}')
        matches.append({
            'round': row['Round'], 'date': row['Date'],
            'time_as_listed': row['Time'],
            'team_a': team_name(row['Home'], 'home'),
            'team_b': team_name(row['Away'], 'away'),
            'team_a_goals': int(goals_a), 'team_b_goals': int(goals_b),
            'goal_difference': int(goals_a) - int(goals_b),
            'shootout_team_a': '' if pen_a is None else int(pen_a),
            'shootout_team_b': '' if pen_b is None else int(pen_b),
            'venue': row['Venue'], 'attendance': row['Attendance'],
            'notes': row['Notes'],
            'source': 'FBref 2026 World Cup Scores & Fixtures (user CSV export)',
        })
    matches.sort(key=lambda r: (r['date'], r['time_as_listed'], r['team_a'], r['team_b']))
    for i, match in enumerate(matches, 1):
        match['match_id'] = f'WC2026_{i:03d}'
    return rows, matches

def write_csv(path, records, fieldnames):
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def main():
    raw, matches = read_source()
    assert len(raw) == 107, f'Expected 107 exported rows, got {len(raw)}'
    assert len(matches) == 104, f'Expected 104 matches, got {len(matches)}'
    assert len({m['match_id'] for m in matches}) == 104
    assert len({(m['date'], m['team_a'], m['team_b']) for m in matches}) == 104
    assert all(m['team_a'] != m['team_b'] for m in matches)
    assert sum(bool(m['shootout_team_a'] != '') for m in matches) == 4
    match_fields = ['match_id','date','time_as_listed','round','team_a','team_b',
                    'team_a_goals','team_b_goals','goal_difference',
                    'shootout_team_a','shootout_team_b','venue','attendance','notes','source']
    write_csv(MATCH_OUT, matches, match_fields)
    team_rows = []
    for m in matches:
        for side in ('a','b'):
            is_a = side == 'a'
            team_rows.append({
                'match_id': m['match_id'], 'date': m['date'],
                'time_as_listed': m['time_as_listed'], 'round': m['round'],
                'team': m['team_a'] if is_a else m['team_b'],
                'opponent': m['team_b'] if is_a else m['team_a'],
                'listed_side': 'A' if is_a else 'B',
                'goals_scored': m['team_a_goals'] if is_a else m['team_b_goals'],
                'goals_conceded': m['team_b_goals'] if is_a else m['team_a_goals'],
                'goal_difference': m['goal_difference'] if is_a else -m['goal_difference'],
                'source': m['source'],
            })
    assert len(team_rows) == 208
    assert all(sum(x['match_id'] == m['match_id'] for x in team_rows) == 2 for m in matches)
    assert sum(r['goals_scored'] for r in team_rows) == sum(m['team_a_goals'] + m['team_b_goals'] for m in matches)
    team_fields = ['match_id','date','time_as_listed','round','team','opponent',
                   'listed_side','goals_scored','goals_conceded','goal_difference','source']
    write_csv(TEAM_OUT, team_rows, team_fields)
    print('VALIDATED: 104 matches; 208 team-match rows; 4 penalty shootouts handled')
    print('Round counts:', dict(Counter(m['round'] for m in matches)))
    print('Unique teams:', len({r['team'] for r in team_rows}))
    print('Created:', MATCH_OUT.name, TEAM_OUT.name)

if __name__ == '__main__':
    main()

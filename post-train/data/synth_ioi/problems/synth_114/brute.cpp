#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 62);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    cin >> q;
    vector<pair<long long, long long>> lines;
    vector<int> pos_in_act;
    vector<int> act;
    for (int t = 1; t <= q; t++) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            long long a, b;
            cin >> a >> b;
            int id = (int)lines.size();
            lines.push_back({a, b});
            pos_in_act.push_back((int)act.size());
            act.push_back(id);
        } else if (tp == 2) {
            int id;
            cin >> id;
            --id;
            int idx = pos_in_act[id];
            int last = act.back();
            act[idx] = last;
            pos_in_act[last] = idx;
            act.pop_back();
            pos_in_act[id] = -1;
        } else {
            long long x;
            cin >> x;
            if (act.empty()) {
                cout << "EMPTY\n";
            } else {
                long long best = NEG_INF;
                for (int id : act) {
                    best = max(best, lines[id].first * x + lines[id].second);
                }
                cout << best << '\n';
            }
        }
    }
    return 0;
}

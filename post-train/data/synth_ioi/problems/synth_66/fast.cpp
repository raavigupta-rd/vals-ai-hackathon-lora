#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    vector<pair<int, long long>> cur, nxt, all_g;
    all_g.reserve(n * 10);
    for (int x : a) {
        nxt.clear();
        for (auto [g, c] : cur) {
            int ng = std::gcd(g, x);
            if (!nxt.empty() && nxt.back().first == ng) nxt.back().second += c;
            else nxt.push_back({ng, c});
        }
        if (!nxt.empty() && nxt.back().first == x) nxt.back().second += 1;
        else nxt.push_back({x, 1});
        cur.swap(nxt);
        for (auto& pr : cur) all_g.push_back(pr);
    }

    sort(all_g.begin(), all_g.end());
    vector<pair<int, long long>> merged;
    merged.reserve(all_g.size());
    for (auto [g, c] : all_g) {
        if (!merged.empty() && merged.back().first == g) merged.back().second += c;
        else merged.push_back({g, c});
    }

    while (q--) {
        int x; cin >> x;
        auto it = lower_bound(merged.begin(), merged.end(), make_pair(x, -1LL));
        if (it != merged.end() && it->first == x) cout << it->second << '\n';
        else cout << 0 << '\n';
    }
    return 0;
}

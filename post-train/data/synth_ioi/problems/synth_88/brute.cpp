#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    set<pair<int, int>> edges;
    vector<int> p(n + 1), sz(n + 1);
    auto findp = [&](auto& self, int x) -> int { return p[x] == x ? x : p[x] = self(self, p[x]); };

    while (q--) {
        int tp; cin >> tp;
        if (tp == 1) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            edges.insert({u, v});
        } else if (tp == 2) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            edges.erase({u, v});
        } else {
            iota(p.begin(), p.end(), 0);
            fill(sz.begin(), sz.end(), 1);
            long long pairs = n;
            for (auto [u, v] : edges) {
                int a = findp(findp, u), b = findp(findp, v);
                if (a != b) {
                    pairs += 2LL * sz[a] * sz[b];
                    p[b] = a;
                    sz[a] += sz[b];
                }
            }
            cout << pairs << '\n';
        }
    }
    return 0;
}

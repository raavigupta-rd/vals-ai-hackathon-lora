#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Exact deterministic AHU tuple-ID table shared across all roots
    map<vector<int>, int> ahu;
    auto get_ahu_id = [&](vector<int>& ch) -> int {
        sort(ch.begin(), ch.end());
        auto [it, ins] = ahu.emplace(ch, (int)ahu.size() + 1);
        return it->second;
    };

    vector<int> par(n + 1), ord(n), sub_id(n + 1), raw_root(n + 1);
    for (int r = 1; r <= n; r++) {
        par[r] = 0;
        int head = 0, tail = 0;
        ord[tail++] = r;
        while (head < tail) {
            int u = ord[head++];
            for (int v : adj[u]) {
                if (v != par[u]) {
                    par[v] = u;
                    ord[tail++] = v;
                }
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            int u = ord[i];
            vector<int> ch;
            for (int v : adj[u]) {
                if (v != par[u]) ch.push_back(sub_id[v]);
            }
            sub_id[u] = get_ahu_id(ch);
        }
        raw_root[r] = sub_id[r];
    }

    map<int, int> remap;
    int cnt = 0;
    vector<int> ans(n + 1);
    for (int u = 1; u <= n; u++) {
        auto [it, ins] = remap.emplace(raw_root[u], cnt + 1);
        if (ins) ++cnt;
        ans[u] = it->second;
    }
    cout << cnt << '\n';
    for (int u = 1; u <= n; u++) cout << ans[u] << (u == n ? '\n' : ' ');
    return 0;
}

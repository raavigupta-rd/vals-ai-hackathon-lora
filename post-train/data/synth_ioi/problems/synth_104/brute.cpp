#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, unsigned long long>>> adj(n + 1);
    vector<int> vis(n + 1, 0), que(n + 1);
    vector<unsigned long long> dist(n + 1, 0);
    int timer = 0;

    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u, v;
            unsigned long long w;
            cin >> u >> v >> w;
            adj[u].push_back({v, w});
            adj[v].push_back({u, w});
        } else {
            int s, t;
            cin >> s >> t;
            ++timer;
            int head = 0, tail = 0;
            vis[s] = timer;
            dist[s] = 0;
            que[tail++] = s;
            unsigned long long b[60]{};
            auto ins = [&](unsigned long long x) {
                for (int i = 59; i >= 0; i--) {
                    if (!((x >> i) & 1ULL)) continue;
                    if (!b[i]) {
                        b[i] = x;
                        return;
                    }
                    x ^= b[i];
                }
            };
            while (head < tail) {
                int u = que[head++];
                for (auto [v, w] : adj[u]) {
                    if (vis[v] != timer) {
                        vis[v] = timer;
                        dist[v] = dist[u] ^ w;
                        que[tail++] = v;
                    } else {
                        ins(dist[u] ^ dist[v] ^ w);
                    }
                }
            }
            if (vis[t] != timer) {
                cout << -1 << '\n';
            } else {
                unsigned long long ans = dist[t];
                for (int i = 59; i >= 0; i--) {
                    if ((ans ^ b[i]) > ans) ans ^= b[i];
                }
                cout << ans << '\n';
            }
        }
    }
    return 0;
}

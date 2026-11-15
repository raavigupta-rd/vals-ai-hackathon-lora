#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

// Memory-safe O(N) RAM solver: binary searches X using N tree BFS traversals per check
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long K;
    cin >> n >> K;
    vector<vector<pair<int, long long>>> adj(n + 1);
    long long max_e = 0;
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        max_e = max(max_e, w);
    }

    vector<int> par(n + 1), que(n + 1);
    vector<long long> dist(n + 1);

    auto count_leq = [&](long long X) -> long long {
        long long cnt = 0;
        for (int s = 1; s <= n; s++) {
            int h = 0, t = 0;
            que[t++] = s;
            par[s] = 0;
            dist[s] = 0;
            while (h < t) {
                int u = que[h++];
                if (u > s && dist[u] <= X) cnt++;
                if (dist[u] >= X) continue;
                for (auto [v, w] : adj[u]) {
                    if (v != par[u] && dist[u] + w <= X) {
                        par[v] = u;
                        dist[v] = dist[u] + w;
                        que[t++] = v;
                    }
                }
            }
        }
        return cnt;
    };

    long long lo = 1, hi = max_e * (n - 1), kth_d = hi;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_leq(mid) >= K) {
            kth_d = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    long long cnt_lt = 0, sum_lt = 0;
    for (int s = 1; s <= n; s++) {
        int h = 0, t = 0;
        que[t++] = s;
        par[s] = 0;
        dist[s] = 0;
        while (h < t) {
            int u = que[h++];
            if (u > s && dist[u] < kth_d) {
                cnt_lt++;
                sum_lt = (sum_lt + dist[u]) % MOD;
            }
            for (auto [v, w] : adj[u]) {
                if (v != par[u]) {
                    par[v] = u;
                    dist[v] = dist[u] + w;
                    que[t++] = v;
                }
            }
        }
    }
    long long ans_sum = (sum_lt + (K - cnt_lt) % MOD * (kth_d % MOD)) % MOD;
    cout << kth_d << ' ' << ans_sum << '\n';
    return 0;
}

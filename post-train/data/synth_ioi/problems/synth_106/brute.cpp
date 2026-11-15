#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

// Memory-safe O(N + M + augmentations) implicit Dinic/Edmonds-Karp:
// Forward edges courier j -> tree edge e have INF capacity so they are walked
// directly along the tree path without allocating O(N * M) edges!
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<vector<pair<int, int>>> adj(n + 1);
    vector<long long> ecap(n, 0);
    for (int i = 1; i <= n - 1; i++) {
        int u, v;
        long long c;
        cin >> u >> v >> c;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
        ecap[i] = c;
    }

    vector<int> par(n + 1, 0), par_edge(n + 1, 0), dep(n + 1, 0), q_bfs(n + 1);
    int qh = 0, qt = 0;
    q_bfs[qt++] = 1;
    while (qh < qt) {
        int u = q_bfs[qh++];
        for (auto [v, id] : adj[u]) {
            if (v != par[u]) {
                par[v] = u;
                par_edge[v] = id;
                dep[v] = dep[u] + 1;
                q_bfs[qt++] = v;
            }
        }
    }

    vector<int> cu(m), cv(m);
    vector<long long> scap(m);
    for (int j = 0; j < m; j++) {
        cin >> cu[j] >> cv[j] >> scap[j];
        if (cu[j] == cv[j]) scap[j] = 0;
    }

    // Sparse positive flow f[e][j] > 0 stored per tree edge e
    vector<unordered_map<int, long long>> rev_flow(n);

    // Node numbering for BFS:
    // 0: S, 1..m: couriers, m+1 .. m+n-1: tree edges 1..n-1, m+n: T
    int S = 0, T = m + n;
    int V = T + 1;
    vector<int> prev_node(V), que(V);
    vector<long long> bneck(V);

    long long total_flow = 0;
    while (true) {
        fill(prev_node.begin(), prev_node.end(), -1);
        int head = 0, tail = 0;
        prev_node[S] = S;
        bneck[S] = INF;
        que[tail++] = S;

        while (head < tail && prev_node[T] == -1) {
            int x = que[head++];
            if (x == S) {
                for (int j = 0; j < m; j++) {
                    if (scap[j] > 0 && prev_node[1 + j] == -1) {
                        prev_node[1 + j] = S;
                        bneck[1 + j] = scap[j];
                        que[tail++] = 1 + j;
                    }
                }
            } else if (x >= 1 && x <= m) {
                int j = x - 1;
                int a = cu[j], b = cv[j];
                while (a != b) {
                    if (dep[a] < dep[b]) swap(a, b);
                    int eid = par_edge[a];
                    int enode = m + eid;
                    if (prev_node[enode] == -1) {
                        prev_node[enode] = x;
                        bneck[enode] = bneck[x];
                        que[tail++] = enode;
                        if (ecap[eid] > 0 && prev_node[T] == -1) {
                            prev_node[T] = enode;
                            bneck[T] = min(bneck[enode], ecap[eid]);
                            break;
                        }
                    }
                    a = par[a];
                }
            } else if (x > m && x < T) {
                int eid = x - m;
                if (ecap[eid] > 0 && prev_node[T] == -1) {
                    prev_node[T] = x;
                    bneck[T] = min(bneck[x], ecap[eid]);
                    break;
                }
                for (const auto& [j, fl] : rev_flow[eid]) {
                    if (fl > 0 && prev_node[1 + j] == -1) {
                        prev_node[1 + j] = x;
                        bneck[1 + j] = min(bneck[x], fl);
                        que[tail++] = 1 + j;
                    }
                }
            }
        }

        if (prev_node[T] == -1) break;
        long long push = bneck[T];
        total_flow += push;

        int cur = T;
        while (cur != S) {
            int p = prev_node[cur];
            if (cur == T) {
                ecap[p - m] -= push;
            } else if (p == S) {
                scap[cur - 1] -= push;
            } else if (p >= 1 && p <= m) {
                int j = p - 1, eid = cur - m;
                rev_flow[eid][j] += push;
            } else {
                int eid = p - m, j = cur - 1;
                auto it = rev_flow[eid].find(j);
                it->second -= push;
                if (it->second == 0) rev_flow[eid].erase(it);
            }
            cur = p;
        }
    }

    cout << total_flow << '\n';
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    int eid_cnt = 0;

    vector<int> tin(n + 1, 0), low(n + 1, 0), comp2(n + 1, 0);
    vector<char> is_bridge;
    long long last = 0;

    while (q--) {
        int tp, rx, ry;
        cin >> tp >> rx >> ry;
        int u = (int)((rx + last) % n) + 1;
        int v = (int)((ry + last) % n) + 1;
        if (tp == 1) {
            int id = eid_cnt++;
            adj[u].push_back({v, id});
            adj[v].push_back({u, id});
        } else {
            is_bridge.assign(eid_cnt, 0);
            // Iterative DFS from u to find component and bridges
            vector<int> visited;
            struct Frame {
                int u, peid, idx;
            };
            vector<Frame> stk;
            int timer = 0, br_cnt = 0;
            tin[u] = low[u] = ++timer;
            visited.push_back(u);
            stk.push_back({u, -1, 0});
            while (!stk.empty()) {
                auto& fr = stk.back();
                int x = fr.u;
                if (fr.idx < (int)adj[x].size()) {
                    auto [y, id] = adj[x][fr.idx++];
                    if (id == fr.peid) continue;
                    if (!tin[y]) {
                        tin[y] = low[y] = ++timer;
                        visited.push_back(y);
                        stk.push_back({y, id, 0});
                    } else {
                        low[x] = min(low[x], tin[y]);
                    }
                } else {
                    int pe = fr.peid;
                    stk.pop_back();
                    if (!stk.empty()) {
                        int p = stk.back().u;
                        low[p] = min(low[p], low[x]);
                        if (low[x] > tin[p]) {
                            is_bridge[pe] = 1;
                            br_cnt++;
                        }
                    }
                }
            }
            if (!tin[v]) {
                cout << "-1 0\n";
            } else {
                // Check if v reachable from u without crossing bridge
                queue<int> bfs;
                bfs.push(u);
                comp2[u] = 1;
                while (!bfs.empty()) {
                    int x = bfs.front();
                    bfs.pop();
                    for (auto [y, id] : adj[x]) {
                        if (!is_bridge[id] && !comp2[y]) {
                            comp2[y] = 1;
                            bfs.push(y);
                        }
                    }
                }
                int same2 = comp2[v] ? 1 : 0;
                cout << same2 << ' ' << br_cnt << '\n';
                last = (last + same2 + br_cnt) % n;
            }
            for (int x : visited) {
                tin[x] = low[x] = comp2[x] = 0;
            }
        }
    }
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string T;
    int q;
    cin >> T >> q;

    vector<array<int, 26>> nxt(1);
    nxt[0].fill(0);
    vector<int> link(1, 0);
    vector<long long> cnt(1, 0);
    vector<int> end_node(q);

    for (int i = 0; i < q; i++) {
        string p;
        cin >> p;
        int u = 0;
        for (char ch : p) {
            int c = ch - 'a';
            if (!nxt[u][c]) {
                nxt[u][c] = (int)nxt.size();
                nxt.emplace_back();
                nxt.back().fill(0);
                link.push_back(0);
                cnt.push_back(0);
            }
            u = nxt[u][c];
        }
        end_node[i] = u;
    }

    vector<int> bfs;
    bfs.reserve(nxt.size());
    for (int c = 0; c < 26; c++) if (nxt[0][c]) bfs.push_back(nxt[0][c]);
    for (size_t i = 0; i < bfs.size(); i++) {
        int u = bfs[i];
        int f = link[u];
        for (int c = 0; c < 26; c++) {
            int v = nxt[u][c];
            if (v) {
                link[v] = nxt[f][c];
                bfs.push_back(v);
            } else {
                nxt[u][c] = nxt[f][c];
            }
        }
    }

    int u = 0;
    for (char ch : T) {
        u = nxt[u][ch - 'a'];
        cnt[u]++;
    }
    for (int i = (int)bfs.size() - 1; i >= 0; i--) {
        int v = bfs[i];
        cnt[link[v]] += cnt[v];
    }
    for (int i = 0; i < q; i++) cout << cnt[end_node[i]] << '\n';
    return 0;
}

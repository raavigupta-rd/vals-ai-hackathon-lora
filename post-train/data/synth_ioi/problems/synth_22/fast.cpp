#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int B = 30;
    int max_nodes = (n + 2) * (B + 1);
    vector<array<int, 2>> ch(max_nodes, {0, 0});
    vector<int> mx_pos(max_nodes, 0);
    vector<int> root(n + 1, 0);
    int ptr = 0;

    auto insert = [&](int prev_root, int val, int idx) {
        int nr = ++ptr;
        int u = nr, p = prev_root;
        mx_pos[u] = idx;
        for (int b = B - 1; b >= 0; b--) {
            int bit = (val >> b) & 1;
            ch[u][bit ^ 1] = ch[p][bit ^ 1];
            int nxt = ++ptr;
            ch[u][bit] = nxt;
            u = nxt;
            p = ch[p][bit];
            mx_pos[u] = idx;
        }
        return nr;
    };

    for (int i = 1; i <= n; i++) {
        int x; cin >> x;
        root[i] = insert(root[i - 1], x, i);
    }

    while (q--) {
        int l, r, x;
        cin >> l >> r >> x;
        int u = root[r], ans = 0;
        for (int b = B - 1; b >= 0; b--) {
            int want = ((x >> b) & 1) ^ 1;
            int c = ch[u][want];
            if (c && mx_pos[c] >= l) {
                ans |= (1 << b);
                u = c;
            } else {
                u = ch[u][want ^ 1];
            }
        }
        cout << ans << '\n';
    }
    return 0;
}

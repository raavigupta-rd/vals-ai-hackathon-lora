#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int K, m, q;
    cin >> K >> m >> q;
    vector<int> reach(K + 2, 0);
    for (int i = 0; i < m; i++) {
        int l, r;
        cin >> l >> r;
        reach[l] = max(reach[l], r + 1);
    }
    for (int x = 1; x <= K + 1; x++) {
        reach[x] = max({reach[x], reach[x - 1], x});
    }
    int LOG = 1;
    while ((1 << LOG) <= K + 2) LOG++;
    vector<vector<int>> up(LOG, vector<int>(K + 2, K + 1));
    for (int x = 1; x <= K + 1; x++) up[0][x] = reach[x];
    for (int b = 1; b < LOG; b++)
        for (int x = 1; x <= K + 1; x++)
            up[b][x] = up[b - 1][up[b - 1][x]];

    while (q--) {
        int a, b;
        cin >> a >> b;
        int target = b + 1;
        if (up[LOG - 1][a] < target) {
            cout << -1 << '\n';
            continue;
        }
        int ans = 0, cur = a;
        for (int k = LOG - 1; k >= 0; k--) {
            if (up[k][cur] < target) {
                ans += (1 << k);
                cur = up[k][cur];
            }
        }
        cout << (ans + 1) << '\n';
    }
    return 0;
}

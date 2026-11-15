#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<int> own(m + 1);
    for (int j = 1; j <= m; j++) cin >> own[j];
    vector<long long> req(n + 1), cur(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> req[i];
    vector<int> ans(n + 1, -1);
    for (int t = 1; t <= q; t++) {
        int l, r; long long w;
        cin >> l >> r >> w;
        for (int j = l; j <= r; j++) {
            int p = own[j];
            if (ans[p] == -1) {
                cur[p] += w;
                if (cur[p] >= req[p]) ans[p] = t;
            }
        }
    }
    for (int i = 1; i <= n; i++) cout << ans[i] << '\n';
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int B = 30;
    vector<array<int, B>> bas(n + 1), pos(n + 1);
    bas[0].fill(0);
    pos[0].fill(0);

    for (int i = 1; i <= n; i++) {
        int x; cin >> x;
        bas[i] = bas[i - 1];
        pos[i] = pos[i - 1];
        int p = i;
        for (int b = B - 1; b >= 0; b--) {
            if ((x >> b) & 1) {
                if (!bas[i][b]) {
                    bas[i][b] = x;
                    pos[i][b] = p;
                    break;
                }
                if (pos[i][b] < p) {
                    swap(bas[i][b], x);
                    swap(pos[i][b], p);
                }
                x ^= bas[i][b];
            }
        }
    }

    while (q--) {
        int l, r;
        cin >> l >> r;
        int ans = 0;
        const auto& bv = bas[r];
        const auto& pv = pos[r];
        for (int b = B - 1; b >= 0; b--) {
            if (pv[b] >= l && (ans ^ bv[b]) > ans) ans ^= bv[b];
        }
        cout << ans << '\n';
    }
    return 0;
}

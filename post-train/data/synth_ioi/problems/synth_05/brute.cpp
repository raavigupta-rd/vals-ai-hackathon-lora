#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i];
    vector<int> cnt(n + 1, 0);
    while (q--) {
        int t, l, r;
        cin >> t >> l >> r;
        int d = (t == 1 ? 1 : -1);
        for (int i = l; i <= r; i++) cnt[i] += d;
        long long ans = 0;
        for (int i = 1; i <= n; i++) if (cnt[i] > 0) ans += w[i];
        cout << ans << '\n';
    }
    return 0;
}

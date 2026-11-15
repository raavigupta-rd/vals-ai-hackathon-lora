#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int l, r, x;
        cin >> l >> r >> x;
        int best = 0;
        for (int i = l; i <= r; i++) best = max(best, a[i] ^ x);
        cout << best << '\n';
    }
    return 0;
}

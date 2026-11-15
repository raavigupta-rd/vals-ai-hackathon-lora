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
        int l, r, x; cin >> l >> r >> x;
        long long s = 0;
        for (int i = l; i <= r; i++) if (a[i] <= x) s += a[i];
        cout << s << '\n';
    }
    return 0;
}

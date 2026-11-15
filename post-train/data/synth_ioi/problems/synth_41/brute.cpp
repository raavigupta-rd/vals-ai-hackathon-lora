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
        int t, x, y;
        cin >> t >> x >> y;
        if (t == 1) a[x] = y;
        else {
            int ans = -1;
            for (int i = x; i <= n; i++) if (a[i] >= y) { ans = i; break; }
            cout << ans << '\n';
        }
    }
    return 0;
}

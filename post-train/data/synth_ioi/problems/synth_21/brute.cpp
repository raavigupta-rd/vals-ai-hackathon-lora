#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int p; char c;
            cin >> p >> c;
            s[p - 1] = c;
        } else {
            int l, r;
            cin >> l >> r;
            --l; --r;
            bool ok = true;
            while (l < r) {
                if (s[l++] != s[r--]) { ok = false; break; }
            }
            cout << (ok ? "YES\n" : "NO\n");
        }
    }
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    while (q--) {
        int l, r;
        cin >> l >> r;
        --l; --r;
        int best = 1;
        for (int c = l; c <= r; c++) {
            int k = 1;
            while (c - k >= l && c + k <= r && s[c - k] == s[c + k]) k++;
            best = max(best, 2 * k - 1);
            int e = 0;
            while (c - e >= l && c + 1 + e <= r && s[c - e] == s[c + 1 + e]) e++;
            best = max(best, 2 * e);
        }
        cout << best << '\n';
    }
    return 0;
}

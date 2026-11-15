#include <bits/stdc++.h>
using namespace std;

struct SparseMax {
    int n, lg;
    vector<vector<int>> st;
    void build(const vector<int>& a) {
        n = (int)a.size();
        if (n == 0) return;
        lg = __lg(n) + 1;
        st.assign(lg, vector<int>(n));
        st[0] = a;
        for (int k = 1; k < lg; k++) {
            int h = 1 << (k - 1);
            for (int i = 0; i + (1 << k) <= n; i++)
                st[k][i] = max(st[k - 1][i], st[k - 1][i + h]);
        }
    }
    inline int query(int l, int r) const {
        int k = __lg(r - l + 1);
        return max(st[k][l], st[k][r - (1 << k) + 1]);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    // d1[i]: max odd radius r >= 1 s.t. s[i-r+1 .. i+r-1] is palindrome
    vector<int> d1(n);
    for (int i = 0, l = 0, r = -1; i < n; i++) {
        int k = (i > r) ? 1 : min(d1[l + r - i], r - i + 1);
        while (0 <= i - k && i + k < n && s[i - k] == s[i + k]) k++;
        d1[i] = k--;
        if (i + k > r) { l = i - k; r = i + k; }
    }

    // d2[i] (0 <= i < n-1): max even radius r >= 0 s.t. s[i-r+1 .. i+r] is palindrome
    vector<int> d2(max(0, n - 1), 0);
    for (int i = 0, l = 0, r = -1; i < n - 1; i++) {
        int k = (i + 1 > r) ? 0 : min(d2[l + r - i - 1], r - i);
        while (0 <= i - k && i + 1 + k < n && s[i - k] == s[i + 1 + k]) k++;
        d2[i] = k;
        if (i + k > r) { l = i - k + 1; r = i + k; }
    }

    SparseMax rmq1, rmq2;
    rmq1.build(d1);
    if (n > 1) rmq2.build(d2);

    while (q--) {
        int l, r;
        cin >> l >> r;
        --l; --r;
        int len = r - l + 1;

        // Odd radius k in [1, (len + 1) / 2]
        int lo = 1, hi = (len + 1) / 2;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (rmq1.query(l + mid - 1, r - mid + 1) >= mid) lo = mid;
            else hi = mid - 1;
        }
        int ans = 2 * lo - 1;

        // Even radius k in [1, len / 2]
        if (len >= 2) {
            int elo = 0, ehi = len / 2;
            while (elo < ehi) {
                int mid = (elo + ehi + 1) >> 1;
                if (rmq2.query(l + mid - 1, r - mid) >= mid) elo = mid;
                else ehi = mid - 1;
            }
            ans = max(ans, 2 * elo);
        }
        cout << ans << '\n';
    }
    return 0;
}

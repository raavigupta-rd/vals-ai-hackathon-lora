#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1), buf;
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int l, r, k;
        cin >> l >> r >> k;
        buf.assign(a.begin() + l, a.begin() + r + 1);
        nth_element(buf.begin(), buf.begin() + (k - 1), buf.end());
        long long kth = buf[k - 1], sum = 0;
        for (int i = 0; i < k; i++) sum += buf[i];
        cout << kth << ' ' << sum << '\n';
    }
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    int mx = 0;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }
    vector<int> cnt(mx + 1, 0), touched;
    while (q--) {
        int l, r; cin >> l >> r;
        touched.clear();
        for (int i = l; i <= r; i++) {
            if (!cnt[a[i]]) touched.push_back(a[i]);
            cnt[a[i]]++;
        }
        long long s = 0;
        for (int x : touched) {
            s += 1LL * cnt[x] * cnt[x] * x;
            cnt[x] = 0;
        }
        cout << s << '\n';
    }
    return 0;
}

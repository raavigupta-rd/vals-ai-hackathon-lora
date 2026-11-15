#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<int> seen(n + 3, 0);
    for (int t = 1; t <= q; t++) {
        int l, r;
        cin >> l >> r;
        for (int i = l; i <= r; i++) if (a[i] <= n + 1) seen[a[i]] = t;
        int mex = 0;
        while (seen[mex] == t) mex++;
        cout << mex << '\n';
    }
    return 0;
}

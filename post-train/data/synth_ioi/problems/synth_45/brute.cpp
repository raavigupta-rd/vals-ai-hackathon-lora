#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<int> l(m), r(m);
    vector<long long> w(m);
    for (int i = 0; i < m; i++) cin >> l[i] >> r[i] >> w[i];
    while (q--) {
        int a, b; cin >> a >> b;
        long long s = 0;
        for (int i = 0; i < m; i++) if (a <= l[i] && r[i] <= b) s += w[i];
        cout << s << '\n';
    }
    return 0;
}

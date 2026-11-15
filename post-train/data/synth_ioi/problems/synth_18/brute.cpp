#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> m(n);
    vector<long long> p(n);
    for (int i = 0; i < n; i++) cin >> m[i] >> p[i];
    while (q--) {
        int x;
        cin >> x;
        long long s = 0;
        for (int i = 0; i < n; i++) if ((m[i] & x) == m[i]) s += p[i];
        cout << s << '\n';
    }
    return 0;
}

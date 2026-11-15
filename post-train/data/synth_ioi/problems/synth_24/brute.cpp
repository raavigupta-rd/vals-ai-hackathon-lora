#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string T;
    int q;
    cin >> T >> q;
    int n = (int)T.size();
    while (q--) {
        string p;
        cin >> p;
        int m = (int)p.size();
        // Naive character scan O(N * M) worst-case, fast on small
        long long ans = 0;
        for (int i = 0; i + m <= n; i++) {
            bool ok = true;
            for (int j = 0; j < m; j++) if (T[i + j] != p[j]) { ok = false; break; }
            if (ok) ans++;
        }
        cout << ans << '\n';
    }
    return 0;
}

#include <bits/stdc++.h>
using namespace std;

// 40-pt subtask solver: per-query Z-algorithm on S[A..B] + '#' + S[L..R] in O(Q * N)
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    string t;
    vector<int> z;
    while (q--) {
        int l, r, a, b;
        cin >> l >> r >> a >> b;
        int plen = b - a + 1, tlen = r - l + 1;
        t.clear();
        t.append(s, a - 1, plen);
        t.push_back('#');
        t.append(s, l - 1, tlen);
        int m = (int)t.size();
        z.assign(m, 0);
        int best = 0;
        for (int i = 1, zl = 0, zr = 0; i < m; i++) {
            if (i <= zr) z[i] = min(zr - i + 1, z[i - zl]);
            while (i + z[i] < m && t[z[i]] == t[i + z[i]]) z[i]++;
            if (i + z[i] - 1 > zr) { zl = i; zr = i + z[i] - 1; }
            if (i > plen) best = max(best, z[i]);
        }
        cout << best << '\n';
    }
    return 0;
}

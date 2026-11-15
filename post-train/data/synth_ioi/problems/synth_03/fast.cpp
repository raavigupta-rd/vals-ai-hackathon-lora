#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> h(n + 1), pref(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        cin >> h[i];
        pref[i] = pref[i - 1] + h[i];
    }

    int LOG = 1;
    while ((1 << LOG) <= n + 1) LOG++;

    // Sparse tables for leftmost max and rightmost max
    vector<vector<int>> stL(LOG, vector<int>(n + 1)), stR(LOG, vector<int>(n + 1));
    for (int i = 1; i <= n; i++) stL[0][i] = stR[0][i] = i;
    for (int k = 1; k < LOG; k++) {
        int len = 1 << (k - 1);
        for (int i = 1; i + (1 << k) - 1 <= n; i++) {
            int a = stL[k - 1][i], b = stL[k - 1][i + len];
            stL[k][i] = (h[a] >= h[b] ? a : b);
            int c = stR[k - 1][i], d = stR[k - 1][i + len];
            stR[k][i] = (h[d] >= h[c] ? d : c);
        }
    }
    auto query_bounds = [&](int l, int r) -> pair<int, int> {
        int k = __lg(r - l + 1);
        int a = stL[k][l], b = stL[k][r - (1 << k) + 1];
        int ml = (h[a] >= h[b] ? a : b);
        int c = stR[k][l], d = stR[k][r - (1 << k) + 1];
        int mr = (h[d] >= h[c] ? d : c);
        return {ml, mr};
    };

    // Next strictly greater to the right (1..n, sentinel n+1)
    vector<int> nxt(n + 2, n + 1);
    {
        vector<int> st;
        for (int i = n; i >= 1; i--) {
            while (!st.empty() && h[st.back()] <= h[i]) st.pop_back();
            nxt[i] = st.empty() ? n + 1 : st.back();
            st.push_back(i);
        }
    }
    // Previous strictly greater to the left (1..n, sentinel 0)
    vector<int> prv(n + 2, 0);
    {
        vector<int> st;
        for (int i = 1; i <= n; i++) {
            while (!st.empty() && h[st.back()] <= h[i]) st.pop_back();
            prv[i] = st.empty() ? 0 : st.back();
            st.push_back(i);
        }
    }

    vector<vector<int>> upR(LOG, vector<int>(n + 2, n + 1));
    vector<vector<long long>> sumR(LOG, vector<long long>(n + 2, 0));
    for (int i = 1; i <= n; i++) {
        upR[0][i] = nxt[i];
        if (nxt[i] <= n) sumR[0][i] = 1LL * (nxt[i] - i) * h[i];
    }
    for (int k = 1; k < LOG; k++) {
        for (int i = 1; i <= n; i++) {
            int mid = upR[k - 1][i];
            upR[k][i] = upR[k - 1][mid];
            sumR[k][i] = sumR[k - 1][i] + sumR[k - 1][mid];
        }
    }

    vector<vector<int>> upL(LOG, vector<int>(n + 2, 0));
    vector<vector<long long>> sumL(LOG, vector<long long>(n + 2, 0));
    for (int i = 1; i <= n; i++) {
        upL[0][i] = prv[i];
        if (prv[i] >= 1) sumL[0][i] = 1LL * (i - prv[i]) * h[i];
    }
    for (int k = 1; k < LOG; k++) {
        for (int i = 1; i <= n; i++) {
            int mid = upL[k - 1][i];
            upL[k][i] = upL[k - 1][mid];
            sumL[k][i] = sumL[k - 1][i] + sumL[k - 1][mid];
        }
    }

    while (q--) {
        int l, r;
        cin >> l >> r;
        auto [ml, mr] = query_bounds(l, r);
        long long total_ceil = 0;
        int u = l;
        for (int k = LOG - 1; k >= 0; k--) {
            if (upR[k][u] <= ml) {
                total_ceil += sumR[k][u];
                u = upR[k][u];
            }
        }
        int v = r;
        for (int k = LOG - 1; k >= 0; k--) {
            if (upL[k][v] >= mr) {
                total_ceil += sumL[k][v];
                v = upL[k][v];
            }
        }
        total_ceil += 1LL * (mr - ml + 1) * h[ml];
        cout << (total_ceil - (pref[r] - pref[l - 1])) << '\n';
    }
    return 0;
}

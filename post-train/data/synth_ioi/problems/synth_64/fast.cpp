#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    // O(N log N) radix / counting Suffix Array
    vector<int> sa(n), rk(n), nrk(n), cnt(max(256, n) + 1, 0), tmp(n);
    for (int i = 0; i < n; i++) cnt[(unsigned char)s[i]]++;
    for (int i = 1; i < 256; i++) cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--) sa[--cnt[(unsigned char)s[i]]] = i;
    rk[sa[0]] = 0;
    int classes = 1;
    for (int i = 1; i < n; i++) {
        if (s[sa[i]] != s[sa[i - 1]]) classes++;
        rk[sa[i]] = classes - 1;
    }
    for (int h = 1; h < n && classes < n; h <<= 1) {
        int p = 0;
        for (int i = n - h; i < n; i++) tmp[p++] = i;
        for (int i = 0; i < n; i++) if (sa[i] >= h) tmp[p++] = sa[i] - h;
        fill(cnt.begin(), cnt.begin() + classes, 0);
        for (int i = 0; i < n; i++) cnt[rk[tmp[i]]]++;
        for (int i = 1; i < classes; i++) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--) sa[--cnt[rk[tmp[i]]]] = tmp[i];
        nrk[sa[0]] = 0;
        classes = 1;
        for (int i = 1; i < n; i++) {
            int u = sa[i], v = sa[i - 1];
            int ru2 = (u + h < n ? rk[u + h] : -1);
            int rv2 = (v + h < n ? rk[v + h] : -1);
            if (rk[u] != rk[v] || ru2 != rv2) classes++;
            nrk[u] = classes - 1;
        }
        swap(rk, nrk);
    }

    // Kasai LCP: lcp[i] = LCP(sa[i], sa[i+1]) for 0 <= i < n-1
    vector<int> lcp(max(0, n - 1), 0);
    for (int i = 0, k = 0; i < n; i++) {
        if (rk[i] == n - 1) { k = 0; continue; }
        int j = sa[rk[i] + 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) k++;
        lcp[rk[i]] = k;
        if (k > 0) k--;
    }

    vector<long long> by_freq(n + 2, 0);
    // Frequency = 1
    for (int i = 0; i < n; i++) {
        int adj_mx = 0;
        if (i > 0) adj_mx = max(adj_mx, lcp[i - 1]);
        if (i + 1 < n) adj_mx = max(adj_mx, lcp[i]);
        by_freq[1] += (n - sa[i]) - adj_mx;
    }

    // Frequency >= 2 via Cartesian intervals (< on left, <= on right)
    int m = n - 1;
    if (m > 0) {
        vector<int> L(m, -1), R(m, m), st;
        st.reserve(m);
        for (int i = 0; i < m; i++) {
            while (!st.empty() && lcp[st.back()] >= lcp[i]) st.pop_back();
            L[i] = st.empty() ? -1 : st.back();
            st.push_back(i);
        }
        st.clear();
        for (int i = m - 1; i >= 0; i--) {
            while (!st.empty() && lcp[st.back()] > lcp[i]) st.pop_back();
            R[i] = st.empty() ? m : st.back();
            st.push_back(i);
        }
        for (int i = 0; i < m; i++) {
            int lo = 0;
            if (L[i] != -1) lo = max(lo, lcp[L[i]]);
            if (R[i] != m) lo = max(lo, lcp[R[i]]);
            if (lcp[i] > lo) {
                int freq = R[i] - L[i];
                by_freq[freq] += (lcp[i] - lo);
            }
        }
    }

    vector<long long> pref(n + 1, 0);
    for (int f = 1; f <= n; f++) pref[f] = pref[f - 1] + by_freq[f];

    while (q--) {
        int a, b;
        cin >> a >> b;
        cout << (pref[b] - pref[a - 1]) << '\n';
    }
    return 0;
}

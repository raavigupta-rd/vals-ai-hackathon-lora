#include <bits/stdc++.h>
using namespace std;

vector<int> build_sa(const string& s) {
    int n = (int)s.size(), alpha = 256;
    vector<int> p(n), c(n), cnt(max(alpha, n), 0);
    for (int i = 0; i < n; i++) cnt[(unsigned char)s[i]]++;
    for (int i = 1; i < alpha; i++) cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--) p[--cnt[(unsigned char)s[i]]] = i;
    c[p[0]] = 0;
    int classes = 1;
    for (int i = 1; i < n; i++) {
        if (s[p[i]] != s[p[i - 1]]) classes++;
        c[p[i]] = classes - 1;
    }
    vector<int> pn(n), cn(n);
    for (int h = 0; (1 << h) < n; h++) {
        for (int i = 0; i < n; i++) {
            pn[i] = p[i] - (1 << h);
            if (pn[i] < 0) pn[i] += n;
        }
        fill(cnt.begin(), cnt.begin() + classes, 0);
        for (int i = 0; i < n; i++) cnt[c[pn[i]]]++;
        for (int i = 1; i < classes; i++) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--) p[--cnt[c[pn[i]]]] = pn[i];
        cn[p[0]] = 0;
        classes = 1;
        for (int i = 1; i < n; i++) {
            pair<int, int> cur = {c[p[i]], c[(p[i] + (1 << h)) % n]};
            pair<int, int> prv = {c[p[i - 1]], c[(p[i - 1] + (1 << h)) % n]};
            if (cur != prv) classes++;
            cn[p[i]] = classes - 1;
        }
        c.swap(cn);
        if (classes == n) break;
    }
    return p;
}

vector<int> build_lcp(const string& s, const vector<int>& p) {
    int n = (int)s.size();
    vector<int> rank(n, 0), lcp(n - 1, 0);
    for (int i = 0; i < n; i++) rank[p[i]] = i;
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (rank[i] == n - 1) { k = 0; continue; }
        int j = p[rank[i] + 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) k++;
        lcp[rank[i]] = k;
        if (k) k--;
    }
    return lcp;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string A, B;
    cin >> A >> B;
    int nA = (int)A.size();
    string S = A + "#" + B + "$";
    auto sa = build_sa(S);
    auto lcp = build_lcp(S, sa);
    int m = (int)S.size();

    auto which = [&](int idx) -> int {
        if (idx < nA) return 0;
        if (idx > nA && idx < m - 1) return 1;
        return -1;
    };

    auto count_dir = [&](int src_type, int dst_type) -> long long {
        long long total = 0, cur_sum = 0;
        vector<pair<int, long long>> st; // (lcp_val, count_of_src)
        for (int i = 0; i < m; i++) {
            int tp = which(sa[i]);
            if (tp == dst_type) total += cur_sum;
            if (i + 1 == m) break;
            long long add_cnt = (tp == src_type ? 1 : 0);
            int h = lcp[i];
            while (!st.empty() && st.back().first >= h) {
                cur_sum -= 1LL * st.back().first * st.back().second;
                add_cnt += st.back().second;
                st.pop_back();
            }
            if (add_cnt > 0) {
                st.push_back({h, add_cnt});
                cur_sum += 1LL * h * add_cnt;
            }
        }
        return total;
    };

    cout << (count_dir(0, 1) + count_dir(1, 0)) << '\n';
    return 0;
}

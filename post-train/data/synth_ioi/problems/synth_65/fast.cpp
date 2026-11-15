#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string t;
    int k;
    cin >> t >> k;

    vector<array<int, 26>> tr;
    vector<int> fail, best_len;
    tr.push_back({});
    tr[0].fill(0);
    fail.push_back(0);
    best_len.push_back(0);

    for (int i = 0; i < k; i++) {
        string p; cin >> p;
        int u = 0;
        for (char ch : p) {
            int c = ch - 'a';
            if (!tr[u][c]) {
                tr[u][c] = (int)tr.size();
                tr.push_back({});
                tr.back().fill(0);
                fail.push_back(0);
                best_len.push_back(0);
            }
            u = tr[u][c];
        }
        best_len[u] = max(best_len[u], (int)p.size());
    }

    queue<int> q;
    for (int c = 0; c < 26; c++) {
        if (tr[0][c]) q.push(tr[0][c]);
    }
    while (!q.empty()) {
        int u = q.front(); q.pop();
        best_len[u] = max(best_len[u], best_len[fail[u]]);
        for (int c = 0; c < 26; c++) {
            int v = tr[u][c];
            if (v) {
                fail[v] = tr[fail[u]][c];
                q.push(v);
            } else {
                tr[u][c] = tr[fail[u]][c];
            }
        }
    }

    string res;
    res.reserve(t.size());
    vector<int> st = {0};
    st.reserve(t.size() + 1);

    for (char ch : t) {
        int u = tr[st.back()][ch - 'a'];
        res.push_back(ch);
        st.push_back(u);
        int rem = best_len[u];
        if (rem > 0) {
            res.resize(res.size() - rem);
            st.resize(st.size() - rem);
        }
    }

    if (res.empty()) cout << "EMPTY\n";
    else cout << res << '\n';
    return 0;
}

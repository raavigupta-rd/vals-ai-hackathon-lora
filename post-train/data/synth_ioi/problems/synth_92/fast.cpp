#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 998244353;

struct Node {
    int nxt[26];
    int len = 0, fail = 0, diff = 0, slink = 0;
    Node() { memset(nxt, 0, sizeof(nxt)); }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string s;
    cin >> s;
    int n = (int)s.size();

    // Node 0: len = -1 (odd root), Node 1: len = 0 (even root)
    vector<Node> tr(2);
    tr.reserve(n + 5);
    tr[0].len = -1; tr[0].fail = 0;
    tr[1].len = 0;  tr[1].fail = 0;
    int last = 1;

    string pad = "#" + s;
    auto get_fail = [&](int u, int pos) {
        while (pad[pos - tr[u].len - 1] != pad[pos]) u = tr[u].fail;
        return u;
    };

    vector<int> dp(n + 1, 0), g(n + 5, 0);
    dp[0] = 1;

    for (int i = 1; i <= n; i++) {
        int c = pad[i] - 'a';
        int cur = get_fail(last, i);
        if (!tr[cur].nxt[c]) {
            int nw = (int)tr.size();
            tr.emplace_back();
            tr[nw].len = tr[cur].len + 2;
            int f = tr[get_fail(tr[cur].fail, i)].nxt[c];
            tr[nw].fail = (tr[nw].len == 1 ? 1 : f);
            tr[nw].diff = tr[nw].len - tr[tr[nw].fail].len;
            tr[nw].slink = (tr[nw].diff == tr[tr[nw].fail].diff)
                           ? tr[tr[nw].fail].slink
                           : tr[nw].fail;
            tr[cur].nxt[c] = nw;
        }
        last = tr[cur].nxt[c];

        long long ways_all = 0;
        for (int v = last; tr[v].len > 0; v = tr[v].slink) {
            g[v] = dp[i - tr[tr[v].slink].len - tr[v].diff];
            if (tr[v].diff == tr[tr[v].fail].diff) {
                g[v] += g[tr[v].fail];
                if (g[v] >= MOD) g[v] -= MOD;
            }
            ways_all += g[v];
        }
        // Subtract the unique length-1 palindromic suffix pad[i..i]
        long long ways_ge2 = (ways_all - dp[i - 1]) % MOD;
        if (ways_ge2 < 0) ways_ge2 += MOD;
        dp[i] = (int)ways_ge2;
    }

    cout << dp[n] << '\n';
    return 0;
}

# Grimoire

This repository serves as a collection of my magic spells and a scratchbook.

 <!-- TOC_START -->

<h2>Table of Contents</h2>
<div class="directory-tree" style="font-family: monospace; white-space: pre; overflow-x: auto;">
<strong>grimoire/</strong><br/>
├──&nbsp;📁&nbsp;competition_scripts/<br/>
│&nbsp;&nbsp;&nbsp;└──&nbsp;📁&nbsp;podRacing/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="competition_scripts/podRacing/AIracer-with-drift-control.py">AIracer-with-drift-control.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="competition_scripts/podRacing/Airacterpodupdate.py">Airacterpodupdate.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="competition_scripts/podRacing/hivemindpodcontrol.py">hivemindpodcontrol.py</a><br/>
├──&nbsp;📁&nbsp;experiments/<br/>
│&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;C/<br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/C/epoll_server.c">epoll_server.c</a><br/>
│&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;biology/<br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📁&nbsp;dna-sequencing/<br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/dna-sequence-file-io.log">dna-sequence-file-io.log</a><br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/dna-sequence-file-io.py">dna-sequence-file-io.py</a><br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/dna-sequencing-match.py">dna-sequencing-match.py</a><br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/dna_advanced_search.py">dna_advanced_search.py</a><br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/dna_io_profile.log">dna_io_profile.log</a><br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/dna_io_profile.py">dna_io_profile.py</a><br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/pattern-extraction.py">pattern-extraction.py</a><br/>
│&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/biology/dna-sequencing/readme.md">readme.md</a><br/>
│&nbsp;&nbsp;&nbsp;└──&nbsp;📁&nbsp;python/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;SIMD/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/python/SIMD/swar_.py">swar_.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;VM/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/python/VM/virtualmachine.py">virtualmachine.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;dns/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/python/dns/domain-checker.py">domain-checker.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;gluecodes/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/python/gluecodes/pull-github.py">pull-github.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;numba/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📁&nbsp;np.mean/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/benchmark-branching.py">benchmark-branching.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/benchmark.py">benchmark.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/debugnpmean.cover">debugnpmean.cover</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/debugnpmean.py">debugnpmean.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/npmean.profile">npmean.profile</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/npmean.trace">npmean.trace</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/npmean.trace.html">npmean.trace.html</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/npmeanjit.ipynb">npmeanjit.ipynb</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/npscalartypes.py">npscalartypes.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/numpy._core._methods.cover">numpy._core._methods.cover</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/numpy._core.fromnumeric.cover">numpy._core.fromnumeric.cover</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/scratchbook.ipynb">scratchbook.ipynb</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/trace_to_html.py">trace_to_html.py</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/python/numba/np.mean/trace_to_html.py.html">trace_to_html.py.html</a><br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;📁&nbsp;skiplist/<br/>
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="experiments/python/skiplist/skiplist.py">skiplist.py</a><br/>
└──&nbsp;📁&nbsp;tools/<br/>
&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;finances/<br/>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="tools/finances/downloadfromgmailwithquery.appscript">downloadfromgmailwithquery.appscript</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;├──&nbsp;📁&nbsp;scripts/<br/>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="tools/scripts/add_journal_entry.sh">add_journal_entry.sh</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="tools/scripts/auto-complete.sh">auto-complete.sh</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="tools/scripts/autocommit.sh">autocommit.sh</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="tools/scripts/gitcommitandpush.sh">gitcommitandpush.sh</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├──&nbsp;📄&nbsp;<a href="tools/scripts/llmscripthelper.sh">llmscripthelper.sh</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="tools/scripts/remove_empty_dirs.sh">remove_empty_dirs.sh</a><br/>
&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;📁&nbsp;self/<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──&nbsp;📄&nbsp;<a href="tools/self/update_toc.py">update_toc.py</a><br/>
</div>

<!-- TOC_END -->
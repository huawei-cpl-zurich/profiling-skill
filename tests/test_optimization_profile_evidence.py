import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import optimization_profile_evidence as evidence


def task(index, name='kernel', context='0'):
    return dict(task_id=f'9:2:{index}:0:{context}', name=name, kind='device',
                iteration_id=f'model_iteration_{index}', start_us=index*10., duration_us=2.)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'kernel_details.csv'
        self.tasks = [task(i) for i in range(10)]
        self.rows = [dict(Name='kernel', **{'Task ID': str(i), 'Stream ID': '2',
                     'Context ID': '0', 'Device_id': '5', 'Duration(us)': '2.000',
                     'aic_mac_ratio': str(i/10), 'aiv_vec_ratio': 'N/A',
                     'aic_mte2_time(us)': '3.000\t', 'cube_utilization(%)': '42.50',
                     'Input Shapes': '"64,64;64,64"'}) for i in range(10)]

    def write(self, rows=None):
        rows = self.rows if rows is None else rows
        with self.path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_real_csv_codec_identity_raw_columns_and_statistics(self):
        self.write()
        result = evidence.pmu_rows(self.path, self.tasks, 5)
        kernel = result['kernels']['kernel']
        self.assertEqual(kernel['task_count'], 10)
        self.assertEqual(kernel['rows'][0]['raw']['Input Shapes'], '"64,64;64,64"')
        self.assertEqual(kernel['rows'][0]['raw']['aic_mte2_time(us)'], '3.000\t')
        self.assertAlmostEqual(kernel['numeric_columns']['aic_mac_ratio']['mean'], .45)
        self.assertEqual(kernel['numeric_columns']['aiv_vec_ratio']['status'], 'unavailable')
        self.assertEqual(kernel['numeric_columns']['aiv_vec_ratio']['samples'], [None]*10)
        self.assertEqual(kernel['numeric_columns']['cube_utilization(%)']['mean'], 42.5)
        self.assertEqual(set(kernel['tasks_per_iteration'].values()), {1})

    def test_unavailable_context_does_not_supply_missing_identity(self):
        self.rows[0]['Context ID'] = 'N/A'
        self.tasks[0]['task_id'] = '9:2:0:0:4294967295'
        self.write()
        self.assertEqual(evidence.pmu_rows(self.path, self.tasks, 5)['kernels']['kernel']['rows'][0]['trace_subtask_id'], '4294967295')
        self.tasks.append(dict(self.tasks[0], task_id='10:2:0:1:7'))
        with self.assertRaisesRegex(evidence.EvidenceError, 'unique'):
            evidence.pmu_rows(self.path, self.tasks, 5)

    def test_context_and_subtask_are_independent_namespaces(self):
        self.rows[0]['Context ID']='7';self.write()
        row=evidence.pmu_rows(self.path,self.tasks,5)['kernels']['kernel']['rows'][0]
        self.assertEqual(row['csv_context_id'],'7')
        self.assertEqual(row['trace_subtask_id'],'0')
        self.tasks[0]['context_id']=7
        self.assertEqual(evidence.pmu_rows(self.path,self.tasks,5)['kernels']['kernel']['task_count'],10)
        self.tasks[0]['context_id']=8
        with self.assertRaisesRegex(evidence.EvidenceError,'unique'):
            evidence.pmu_rows(self.path,self.tasks,5)

    def test_duplicate_unmatched_missing_and_device_fail_closed(self):
        for change in ('duplicate', 'unmatched', 'missing', 'device'):
            with self.subTest(change=change):
                rows = copy.deepcopy(self.rows)
                if change == 'duplicate': rows.append(rows[0])
                if change == 'unmatched': rows[0]['Task ID'] = '999'
                if change == 'missing': rows.pop()
                if change == 'device': rows[0]['Device_id'] = '4'
                self.write(rows)
                with self.assertRaises(evidence.EvidenceError):
                    evidence.pmu_rows(self.path, self.tasks, 5)

    def test_partial_metric_is_not_zero_or_complete(self):
        self.rows[1]['aic_mac_ratio'] = 'N/A'
        self.write()
        field = evidence.pmu_rows(self.path, self.tasks, 5)['kernels']['kernel']['numeric_columns']['aic_mac_ratio']
        self.assertEqual(field['status'], 'partial')
        self.assertEqual(field['available_count'], 9)
        self.assertIsNone(field['samples'][1])
        self.rows[1]['aic_mac_ratio'] = 'NaN'
        self.write()
        with self.assertRaisesRegex(evidence.EvidenceError, 'nonfinite'):
            evidence.pmu_rows(self.path, self.tasks, 5)

    def test_present_pmu_headers_with_all_na_values_are_unavailable(self):
        for row in self.rows:
            for column in row:
                if column.startswith(('aic_','aiv_')) or column=='cube_utilization(%)':
                    row[column]='N/A'
        self.write()
        result=evidence.pmu_rows(self.path,self.tasks,5)
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['kernels']['kernel']['task_count'],10)
        self.assertEqual(result['kernels']['kernel']['numeric_columns']['Duration(us)']['mean'],2.)

    def test_timing_samples_and_json_nonfinite_rejected(self):
        self.assertEqual(evidence.samples([1., 3.])['mean'], 2.)
        self.assertAlmostEqual(evidence.samples([1., 3.])['sample_sd'], 2**.5)
        for values in ([True], [float('nan')], [float('inf')], [0.]):
            with self.assertRaises(evidence.EvidenceError):
                evidence.samples(values, positive=True)
        bad = Path(self.temp.name) / 'bad.json'
        bad.write_text('{"bad":NaN}')
        with self.assertRaises(evidence.EvidenceError): evidence.load(bad)

    def test_device_span_includes_interkernel_gaps_without_summing_ratios(self):
        a,b=task(0),task(1)
        a.update(iteration_id='round',start_us=0.,duration_us=2.)
        b.update(iteration_id='round',start_us=10.,duration_us=3.)
        report=dict(tasks=[a,b],iterations=[dict(iteration_id='round',device_active_union_us=5.)])
        row=evidence.operation_windows(report)[0]
        self.assertEqual(row['device_operation_span_us'],13.)
        self.assertEqual(row['device_idle_gap_us'],8.)
        self.assertEqual(row['operation_active_union_us'],5.)
        cpu=dict(a,task_id='cpu',kind='aicpu',start_us=5.,duration_us=2.)
        report['tasks'].append(cpu)
        self.assertEqual(evidence.operation_windows(report)[0]['device_idle_gap_us'],6.)
        report['tasks'].pop()
        b['start_us']=1.;report['iterations'][0]['device_active_union_us']=4.
        self.assertEqual(evidence.operation_windows(report)[0]['device_idle_gap_us'],0.)


class PairTests(unittest.TestCase):
    def fixture(self, root):
        # The actual captured native event fixture exercises normalization, not
        # a self-asserted accounting result or a mocked collector boundary.
        fixture = Path(__file__).parent / 'fixtures/native_flow/after.json'
        original = json.loads(fixture.read_text())['traceEvents']
        events=[]
        for index in range(10):
            selected=copy.deepcopy(original)
            for event in selected:
                event['ts']=str(evidence.Decimal(event['ts'])+index*10000)
                if event['name']=='model_iteration_0':event['name']=f'model_iteration_{index}'
                args=event.get('args',{})
                if 'Task Id' in args:args['Task Id']+=index*100
                if 'connection_id' in args:args['connection_id']+=index*100
                if event.get('cat')=='HostToDevice':
                    event['id']=str(int(event['id'])+(index*100<<32))
                    event['name']='HostToDevice'+event['id']
            events.extend(selected)
        profile = root / 'vendor'; profile.mkdir()
        trace = profile / 'trace_view.json'
        trace.write_text(json.dumps(events))
        normalized = evidence.normalize_trace(events, [f'model_iteration_{i}' for i in range(10)], export_completed=True)
        accounted = evidence.account_profile(normalized['iterations'], normalized['tasks'], capture_complete=True)
        normpath, accpath = root/'normalized.json', root/'accounting.json'
        normpath.write_text(json.dumps(normalized)); accpath.write_text(json.dumps(accounted))
        capture = dict(profile=dict(export_status='complete',vendor_trace=str(trace),trace_sha256=evidence.sha(trace),
                                   capture_path=str(normpath),accounting_path=str(accpath)),
                       forward_calls=10,event_latency=dict(count=10,samples_ms=[.1]*10,instrumentation='profiler active'),
                       kernel_identity=evidence.kernel_identity(accounted,['qualified_native_matmul']),
                       artifacts={str(p):evidence.sha(p) for p in (trace,normpath,accpath)})
        source=root/'kernel.py';source.write_text('retained kernel source')
        compiled=[dict(hash='a'*64,artifacts={'npubin':'b'*64})]
        row = dict(status='complete',correctness=dict(status='complete',calls=1),profiled_calls=10,
                   capture_calls=1,capture=capture,
                   compilation_state_before_capture=compiled,compilation_state_after_capture=copy.deepcopy(compiled),
                   sources={str(source):evidence.sha(source)},metadata={})
        report = dict(protocol='optimization-pair-v1',status='complete',case={'name':'fixture'},fixture={'sha256':'oracle-digest'},
                      counts=dict(correctness_calls=2,capture_calls=2,profiled_calls=20),
                      warmup_calls=0,discovery_calls=0,unprofiled_timing_calls=0,
                      provenance={'physical_die':5},arms={'before':row,'after':copy.deepcopy(row)})
        path = root/'result.json';path.write_text(json.dumps(report))
        return path, report

    def proof(self, root, path, report):
        analysis=root/'analysis';analysis.mkdir(exist_ok=True)
        proof={'original_pair_sha256':evidence.sha(path),'arms':{}}
        for arm in ('before','after'):
            cap=report['arms'][arm]['capture']
            for name,key in (('normalized','capture_path'),('accounting','accounting_path')):
                (analysis/f'{arm}-{name}.json').write_text(Path(cap['profile'][key]).read_text())
            proof['arms'][arm]=dict(status='complete',trace_sha256=cap['profile']['trace_sha256'],capture_forward_calls=10,
                normalized_sha256=evidence.sha(analysis/f'{arm}-normalized.json'),
                accounting_sha256=evidence.sha(analysis/f'{arm}-accounting.json'),kernel_identity=cap['kernel_identity'])
        proofpath=analysis/'reanalysis.json';proofpath.write_text(json.dumps(proof))
        return proofpath,proof

    def test_pair_recomputes_real_flow_proof_and_keeps_missing_pmu_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            path, report = self.fixture(Path(folder))
            result = evidence.collect_pair(path)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['new_device_forwards'],0)
            arm = result['arms']['before']
            self.assertEqual(len(arm['tasks']),10)
            self.assertEqual(arm['metrics']['status'],'unavailable')
            self.assertEqual(arm['sources'],report['arms']['before']['sources'])
            self.assertEqual(arm['event_latency_ms']['count'],10)
            self.assertEqual(arm['device_work_us']['device_busy_us']['count'],10)
            self.assertEqual(result['event_mean_ratio_before_over_after'],1.)

    def test_changed_trace_or_accounting_rejected(self):
        for artifact in ('vendor_trace','accounting_path'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as folder:
                path, report = self.fixture(Path(folder))
                selected = Path(report['arms']['before']['capture']['profile'][artifact])
                selected.write_text('{}')
                with self.assertRaisesRegex(evidence.EvidenceError,'digest'):
                    evidence.collect_pair(path)

    def test_additional_execution_cannot_be_reported_as_single_check_ten_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            path,report=self.fixture(Path(folder))
            report['discovery_calls']=1;path.write_text(json.dumps(report))
            with self.assertRaisesRegex(evidence.EvidenceError,'additional execution'):
                evidence.collect_pair(path)

    def test_reanalysis_bound_to_original_and_retained_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,report=self.fixture(root)
            proofpath,proof=self.proof(root,path,report);analysis=proofpath.parent
            self.assertEqual(evidence.collect_pair(path,reanalysis=proofpath)['status'],'complete')
            for arm in ('before','after'):
                for name in ('normalized','accounting'):
                    (analysis/f'case-{arm}-{name}.json').write_text((analysis/f'{arm}-{name}.json').read_text())
            proofpath.write_text(json.dumps({'cases':{'case':proof}}))
            self.assertEqual(evidence.collect_pair(path,reanalysis=proofpath)['reanalysis_case'],'case')
            proof['original_pair_sha256']='different';proofpath.write_text(json.dumps(proof))
            with self.assertRaisesRegex(evidence.EvidenceError,'different original'):
                evidence.collect_pair(path,reanalysis=proofpath)

    def test_compilation_and_source_drift_rejected_even_with_bound_reanalysis(self):
        for failure in ('changed','missing','invalid','source','compiled-source','binary-metadata'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);path,report=self.fixture(root);row=report['arms']['after']
                if failure=='changed':row['compilation_state_after_capture'][0]['artifacts']['npubin']='c'*64
                if failure=='missing':row.pop('compilation_state_before_capture')
                if failure=='invalid':
                    row['compilation_state_before_capture']=row['compilation_state_after_capture']=[{'hash':'x','binaries':{}}]
                if failure=='source':Path(next(iter(row['sources']))).write_text('changed source')
                if failure in ('compiled-source','binary-metadata'):
                    state=dict(binary_sha256='b'*64,source_sha256=dict(row['sources']))
                    row['compilation_state_before_capture']=row['compilation_state_after_capture']=state
                    if failure=='compiled-source':state['source_sha256']={str(root/'missing.cpp'):'c'*64}
                    else:row['metadata']['binary_sha256']='c'*64
                path.write_text(json.dumps(report));proof,_=self.proof(root,path,report)
                for recovery in (None,proof):
                    with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path,reanalysis=recovery)

    def test_supplied_empty_or_malformed_reanalysis_never_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,report=self.fixture(root);proof=root/'malformed.json'
            for value in ({},None,[],{'cases':[]},{'cases':{}},{'cases':{'case':None}},
                          {'original_pair_sha256':evidence.sha(path)},
                          {'original_pair_sha256':evidence.sha(path),'arms':{'before':{}}}):
                with self.subTest(value=value):
                    proof.write_text(json.dumps(value))
                    with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path,reanalysis=proof)

    def test_reanalysis_only_recovers_known_parser_failures(self):
        for message in ('unvalidated device task type: EVENT_RECORD',
                        'device tasks lack explicit CPU/NPU correlation coverage'):
            with self.subTest(message=message),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);path,report=self.fixture(root);row=report['arms']['after']
                report['status']=row['status']='failed';row['error']=dict(type='CaptureError',message=message)
                path.write_text(json.dumps(report));proof,_=self.proof(root,path,report)
                self.assertEqual(evidence.collect_pair(path,reanalysis=proof)['status'],'complete')
                with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path)

    def test_reanalysis_preserves_other_experiment_failures(self):
        for failure in ('compile','source','cleanup','top-error','unexplained','counts','correctness','extra-capture'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);path,report=self.fixture(root);row=report['arms']['after']
                report['status']=row['status']='failed'
                row['error']=dict(type='CaptureError',message='unvalidated device task type: EVENT_RECORD')
                if failure in ('compile','source'):
                    row['error']['message']='compilation detected during active profiler replay' if failure=='compile' else 'prepared source or binary changed during experiment'
                if failure=='cleanup':row['cleanup']={'status':'failed','reason':'cleanup failed'}
                if failure=='top-error':report['error']={'type':'ValueError','message':'source drift'}
                if failure=='unexplained':row.pop('error')
                if failure=='counts':report['counts']['profiled_calls']=21
                if failure=='correctness':row['correctness']['diagnostics']={'passed':False}
                if failure=='extra-capture':row['capture_calls']=2
                path.write_text(json.dumps(report));proof,_=self.proof(root,path,report)
                with self.assertRaises(evidence.EvidenceError):evidence.collect_pair(path,reanalysis=proof)


if __name__ == '__main__': unittest.main()

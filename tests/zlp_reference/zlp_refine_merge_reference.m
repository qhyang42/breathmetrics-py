% zlp_refine_merge_reference.m - MATLAB reference for tests/test_zlp_refine_merge.py.
%
% Run with the MATLAB toolbox (breathmetrics_functions/, with the ZLP
% refineInhaleOnsets_zlp, mergeIncompleteExhales_zlp and zlpEstimateAllFeatures
% functions) on the path. Reads trace.bin (float64, written by
% make_zlp_refine_merge_reference.py trace) and writes ref.mat. All landmark
% samples in ref.mat are ONE-BASED (MATLAB); the Python converter keeps them
% as they are and the tests subtract 1.
set(0, 'DefaultFigureVisible', 'off');
binFile = 'trace.bin'; fs = 100; outFile = 'ref.mat';
fid = fopen(binFile, 'r'); resp = fread(fid, inf, 'float64', 0, 'ieee-le')'; fclose(fid);
[det, pk0, tr0] = prepBreathTrace_zlp(resp, fs, 'conservative', [], [], 0.05);
[engOn, engPk, engTr] = findInhaleOnsets_zlp(det, fs, pk0, tr0, 'kneeBacktrack', 0.4, 1.25, 0.50, 0.10);

R = struct();
R.fs = fs;
R.engOn = engOn; R.engPk = engPk; R.engTr = engTr;

% refinement, defaults and one option variant
[refOn, ~, ~, refInfo] = refineInhaleOnsets_zlp(resp, det, fs, engOn, engPk, engTr);
R.refOn = refOn; R.refRule = strjoin(cellfun(@(c) ['<' c '>'], refInfo.rule, 'UniformOutput', false), '');
R.refShiftMs = refInfo.shiftMs;
[refOn2, ~, ~, refInfo2] = refineInhaleOnsets_zlp(resp, det, fs, engOn, engPk, engTr, ...
    struct('levelGuard', 'none', 'levelTroughFrac', 0));
R.refVarOn = refOn2; R.refVarRule = strjoin(cellfun(@(c) ['<' c '>'], refInfo2.rule, 'UniformOutput', false), '');

% driver landmarks: stages off / refine / refine + merge
bm0 = zlpEstimateAllFeatures(resp, fs, 'refineOnsets', false);
bm1 = zlpEstimateAllFeatures(resp, fs);
[bm2, ~, z2] = zlpEstimateAllFeatures(resp, fs, 'mergeIncompleteExhales', true);
R.drv0On = bm0.inhaleOnsets; R.drv0Pk = bm0.inhalePeaks; R.drv0Tr = bm0.exhaleTroughs;
R.drv1On = bm1.inhaleOnsets; R.drv1Pk = bm1.inhalePeaks; R.drv1Tr = bm1.exhaleTroughs;
R.drv2On = bm2.inhaleOnsets; R.drv2Pk = bm2.inhalePeaks; R.drv2Tr = bm2.exhaleTroughs;
R.nTroughGuard = z2.merge.nTroughGuard;

% merge on the refined paired landmarks of the driver
n = numel(bm1.inhaleOnsets);
mIn = bm1.inhaleOnsets; mPk = bm1.inhalePeaks(1:n-1); mTr = bm1.exhaleTroughs(1:n-1);
R.mergeInOn = mIn; R.mergeInPk = mPk; R.mergeInTr = mTr;
for v = 1:2
    if v == 1, args = {0.20, struct()}; else, args = {0.5, struct()}; end
    [mg, mi] = mergeIncompleteExhales_zlp(resp, fs, mIn, mPk, mTr, args{:});
    p = sprintf('mg%d', v);
    R.([p 'On']) = mg.onsets; R.([p 'End']) = mg.ends; R.([p 'Pk']) = mg.peaks; R.([p 'Tr']) = mg.troughs;
    R.([p 'Kept']) = mg.keptRow; R.([p 'NPulses']) = mg.nPulses; R.([p 'NRows']) = mg.nRows;
    R.([p 'InArea']) = mg.inhaleArea; R.([p 'ExArea']) = mg.exhaleArea;
    R.([p 'OrigCount']) = cellfun(@numel, mg.origRows); R.([p 'OrigFlat']) = [mg.origRows{:}];
    R.([p 'IsMinor']) = double(mi.isMinor); R.([p 'IsLeadBump']) = double(mi.isLeadBump);
    R.([p 'IsTail']) = double(mi.isTail); R.([p 'IsTroughHead']) = double(mi.isTroughHead);
    R.([p 'Counts']) = [mi.nBefore mi.nAfter mi.nMinor mi.nLeadBumps mi.nMerged mi.nTroughHeads];
end
R.refCounts = [nnz(strcmp(refInfo.rule, 'foot')) nnz(startsWith(refInfo.rule, 'far')) ...
    nnz(startsWith(refInfo.rule, 'level')) nnz(contains(refInfo.rule, 'clamp'))];
save(outFile, '-struct', 'R', '-v7');
fprintf('GENREF fs=%g n=%d eng=%d refined: foot %d far %d level %d clamp %d | merge %d -> %d (minor %d lead %d merged %d heads %d) | final %d\n', ...
    fs, numel(resp), numel(engOn), R.refCounts, R.mg1Counts, numel(bm2.inhaleOnsets));

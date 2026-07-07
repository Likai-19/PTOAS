const fs = require('fs');
const path = require('path');
const readline = require('readline');

// Read file list
const data = fs.readFileSync('C:/Users/53284/.claude/projects/C--Users-53284-Desktop-ptocode/f4152e2f-41d2-4fa0-9c0c-85f55a8d92e7/tool-results/b83s7l7d4.txt', 'utf8');
const filenames = data.trim().split('\n').map(f => {
    const base = path.basename(f);
    return base.replace('.pto', '');
});

// Op name extraction patterns - longer stems first
const opMap = [
    ['active_prefix_index', 'vmi.active_prefix_index'],
    ['create_group_mask', 'vmi.create_group_mask'],
    ['create_mask', 'vmi.create_mask'],
    ['constant_mask', 'vmi.constant_mask'],
    ['constant_', 'vmi.constant'],
    ['channel_merge', 'vmi.channel_merge'],
    ['channel_split', 'vmi.channel_split'],
    ['group_broadcast', 'vmi.group_broadcast'],
    ['group_load', 'vmi.group_load'],
    ['group_reduce', 'vmi.group_reduce'],
    ['group_slot', 'vmi.group_slot_load/vmi.group_slot_store'],
    ['group_store', 'vmi.group_store'],
    ['group_slots', 'vmi.group_slots'],
    ['group_ops', 'vmi.group_broadcast/vmi.group_load/vmi.group_reduce/vmi.group_store'],
    ['masked_load', 'vmi.masked_load'],
    ['masked_store', 'vmi.masked_store'],
    ['mask_logic', 'vmi.mask_and/vmi.mask_or/vmi.mask_xor/vmi.mask_not'],
    ['mask_granularity', 'vmi.ensure_layout(mask)'],
    ['mask_pred', 'vmi.constant_mask'],
    ['mask_concrete', 'vmi.constant_mask'],
    ['mask_select', 'vmi.mask_select'],
    ['mask_use_ensure', 'vmi.ensure_layout'],
    ['mask_remat', 'rematerialize(mask)'],
    ['mask', 'vmi.constant_mask/vmi.create_mask'],
    ['interleaved_memory_ops', 'vmi.load/vmi.store'],
    ['producer_boundary', 'producer_boundary_check'],
    ['stride_load', 'vmi.stride_load'],
    ['stride_store', 'vmi.stride_store'],
    ['expand_load', 'vmi.expand_load'],
    ['call_boundary', 'vmi.call'],
    ['call_argument_boundary', 'vmi.call'],
    ['multi_return', 'vmi.call(multi-return)'],
    ['cf_branch', 'vmi.cf_branch'],
    ['cf_switch', 'vmi.cf_switch'],
    ['scf_for', 'vmi.scf_for'],
    ['scf_if', 'vmi.scf_if'],
    ['scf_while', 'vmi.scf_while'],
    ['scf_execute_region', 'vmi.scf_execute_region'],
    ['scf_index_switch', 'vmi.scf_index_switch'],
    ['load_truncf', 'vmi.load/vmi.truncf'],
    ['load_deint', 'vmi.load(deint)'],
    ['load_safe_tail', 'vmi.load(tail)'],
    ['load_store', 'vmi.load/vmi.store'],
    ['load', 'vmi.load'],
    ['store_deint', 'vmi.store(deint)'],
    ['store_tail', 'vmi.store(tail)'],
    ['store_ensure', 'vmi.store/vmi.ensure_layout'],
    ['store_reduce', 'vmi.store/vmi.reduce'],
    ['store', 'vmi.store'],
    ['sub_mul', 'vmi.subf/vmi.mulf'],
    ['subf', 'vmi.subf'],
    ['mulf', 'vmi.mulf'],
    ['addf', 'vmi.addf'],
    ['addi', 'vmi.addi'],
    ['absf', 'vmi.absf'],
    ['absi', 'vmi.absi'],
    ['negf', 'vmi.negf'],
    ['divf', 'vmi.divf'],
    ['min_max', 'vmi.minf/vmi.mini/vmi.maxf/vmi.maxi'],
    ['minmaxf', 'vmi.minf/vmi.maxf'],
    ['fma', 'vmi.fma'],
    ['extf', 'vmi.extf'],
    ['truncf', 'vmi.truncf'],
    ['trunci', 'vmi.trunci'],
    ['bitcast', 'vmi.bitcast'],
    ['bitwise', 'vmi.bitwise_and/vmi.bitwise_or/vmi.bitwise_xor'],
    ['broadcast', 'vmi.broadcast'],
    ['compress', 'vmi.compress'],
    ['dhist', 'vmi.dhist'],
    ['gather', 'vmi.gather'],
    ['scatter', 'vmi.scatter'],
    ['shuffle', 'vmi.shuffle'],
    ['shli', 'vmi.shli'],
    ['shrui', 'vmi.shrui'],
    ['iota', 'vmi.iota'],
    ['ensure_layout', 'vmi.ensure_layout'],
    ['ensure_mask', 'vmi.ensure_layout(mask)'],
    ['ensure_identity', 'vmi.ensure_layout(identity)'],
    ['elementwise', 'vmi.elementwise(various)'],
    ['unary_math', 'vmi.math.unary'],
    ['unpack', 'vmi.unpack'],
    ['pack', 'vmi.pack'],
    ['quant', 'vmi.quant/vmi.dequant'],
    ['reduce', 'vmi.reduce'],
    ['select', 'vmi.select'],
    ['widen', 'vmi.widen'],
    ['cmp', 'vmi.cmpf/vmi.cmpi'],
    ['integer_cast', 'vmi.trunci/vmi.extui/vmi.extsi'],
    ['type_attr', 'type_attr'],
    ['type_arity', 'vmi.elementwise'],
    ['type_only', 'vmi.elementwise(no-value)'],
    ['type_element', 'vmi.type'],
    ['surface', 'vmi.surface'],
    ['vdintlv', 'vmi.ensure_layout(vdintlv)'],
    ['vintlv', 'vmi.ensure_layout(vintlv)'],
    ['deint', 'vmi.load/vmi.store(deint)'],
    ['deint4', 'vmi.ensure_layout(deint4)'],
    ['rematerialize', 'rematerialize'],
    ['remat', 'rematerialize'],
    ['sink_materialization', 'sink_materialization'],
    ['fold', 'layout_fold'],
    ['legalize', 'legalize'],
    ['memory_space', 'vmi.load/vmi.store(memory_space)'],
    ['memref', 'vmi.load/vmi.store(memref)'],
    ['memory', 'vmi.load/vmi.store'],
    ['chist', 'vmi.chist'],
    ['compaction', 'vmi.compaction'],
    ['construction', 'vpi.construction'],
    ['function_type', 'function'],
    ['backend', 'ptoas_backend'],
    ['private_call', 'vmi.call(private)'],
    ['public_abi', 'vmi.call(public-abi)'],
    ['cli', 'ptoas-cli'],
    ['backend_required', 'ptoas_backend'],
    ['call_inline', 'vmi.call(inline)'],
    ['ento', 'vmi.ento'],
    ['unsupported_op', 'unsupported'],
    ['e2e', 'e2e'],
    ['stable_gather', 'vmi.stable_gather'],
    ['vector_scope', 'vectscope'],
    ['licm', 'licm'],
    ['pipeline', 'pipeline'],
    ['control_flow', 'control_flow'],
    ['external_call', 'vmi.call(external)'],
    ['external_decl', 'vmi.call(external_decl)'],
    ['indirect_call', 'vmi.call(indirect)'],
    ['non_vmi_op', 'non-vmi-op'],
    ['non_load', 'vmi.reduce'],
    ['helper', 'helper'],
    ['shape', 'shape'],
    ['packet', 'packet'],
    ['factor', 'factor'],
    ['basic', 'basic'],
    ['layout_free', 'layout_free'],
    ['residual', 'residual'],
    ['nested', 'nested'],
    ['parse', 'parse'],
    ['shrui_signed', 'vmi.shrui'],
    ['shli_float', 'vmi.shli'],
    ['shrui_float', 'vmi.shrui'],
    ['reassoc', 'vmi.reduce(reassoc)'],
    ['rounding', 'vmi.truncf(rounding)'],
    ['lane_mismatch', 'vmi.addf/vmi.extf/vmi.truncf'],
    ['lane_stride', 'vmi.trunci'],
    ['indices', 'vmi.gather/vmi.scatter/vmi.shuffle'],
    ['kind', 'vmi.elementwise(kind)'],
    ['count', 'vmi.channel_merge/vmi.channel_split/vmi.constant'],
    ['arity', 'vmi.pack/vmi.unpack'],
    ['direction', 'vmi.extf/vmi.truncf'],
    ['integer', 'integer_type_mismatch'],
    ['float', 'float_type_mismatch'],
    ['support', 'hardware_support'],
    ['width', 'construction_width'],
    ['footprint', 'bitcast_footprint'],
    ['partial', 'partial_operation'],
    ['contiguous', 'contiguous_layout'],
    ['safe', 'safe_tail'],
    ['negative_offset', 'negative_offset'],
    ['runtime', 'runtime_mask'],
    ['all_active', 'all_active_mask'],
    ['identity', 'identity'],
    ['dual_layout', 'dual_layout'],
    ['fanout', 'fanout'],
    ['tail', 'tail_handling'],
    ['multi_chunk', 'multi_chunk'],
    ['multichunk', 'multi_chunk'],
    ['conflict', 'layout_conflict'],
    ['granularity', 'granularity'],
    ['unaligned', 'unaligned_stride'],
    ['compact', 'compact_stride'],
    ['unit_stride', 'unit_stride'],
    ['dynamic_stride', 'dynamic_stride'],
    ['non_prefix', 'non_prefix_mask'],
    ['plt_fallback', 'plt_fallback'],
    ['non_splat', 'non_splat'],
    ['x2_widths', 'x2_widths'],
    ['preserve', 'preserve'],
    ['post_gate', 'post_gate'],
    ['gated', 'gated'],
    ['boundary', 'boundary'],
    ['surface', 'surface'],
    ['physical', 'physical'],
    ['b16', 'e2b_b16'],
    ['slots8', 'slots8'],
    ['slots1', 'slots1'],
    ['slots', 'group_slots'],
    ['s256', 's256_region'],
    ['s64', 's64_region'],
    ['s32', 's32_region'],
    ['s16', 's16_region'],
    ['s12', 's12_region'],
    ['f16', 'f16'],
    ['f32', 'f32'],
    ['f8', 'f8'],
    ['fp8', 'fp8'],
    ['bf16', 'bf16'],
    ['i16', 'i16'],
    ['i8', 'i8'],
    ['block8', 'block8'],
    ['dense', 'dense_users'],
    ['typed', 'typed_reduce'],
    ['forwarding', 'shuffle_forwarding'],
    ['splat', 'lane0_splat'],
    ['multitile', 'multitile'],
    ['full_tile', 'full_tile'],
    ['stride', 'stride'],
];

function extractOps(name) {
    const ops = new Set();
    // Remove category prefixes
    const prefixes = [
        'vmi_to_vpto_', 'vmi_layout_assignment_', 'vmi_layout_gate_',
        'vmi_layout_fold_', 'vmi_layout_rematerialize_', 'vmi_layout_sink_materialization_',
        'vmi_layout_group_slots_', 'vmi_layout_factor_',
        'vmi_ptoas_', 'vmi_legalize_', 'vmi_producer_boundary_'
    ];
    let stripped = name;
    for (const p of prefixes) {
        if (stripped.startsWith(p)) {
            stripped = stripped.substring(p.length);
            break;
        }
    }
    if (!prefixes.some(p => name.startsWith(p)) && stripped.startsWith('vmi_')) {
        stripped = stripped.substring(4);
    }

    // Remove _invalid, _valid suffixes
    stripped = stripped.replace(/_(invalid|valid|unsupported|todo)$/, '');

    // Match op stems - long first
    const sortedOps = [...opMap].sort((a, b) => b[0].length - a[0].length);
    for (const [stem, opname] of sortedOps) {
        if (stripped.includes(stem)) {
            ops.add(opname);
        }
    }

    if (ops.size === 0) return ['unknown'];
    return [...ops].sort();
}

function categorize(name) {
    if (name.startsWith('vmi_to_vpto_')) return 'lowering';
    if (name.startsWith('vmi_layout_assignment_')) return 'layout';
    if (name.startsWith('vmi_layout_gate_')) return 'gate';
    if (name.startsWith('vmi_layout_fold_')) return 'layout';
    if (name.startsWith('vmi_layout_rematerialize_')) return 'layout';
    if (name.startsWith('vmi_layout_sink_')) return 'layout';
    if (name.startsWith('vmi_layout_group_slots_')) return 'layout';
    if (name.startsWith('vmi_layout_factor_')) return 'layout';
    if (name.startsWith('vmi_ptoas_')) return 'other';
    if (name.startsWith('vmi_legalize_')) return 'other';
    if (name.startsWith('vmi_producer_boundary_')) return 'other';
    if (name.startsWith('vmi_interleaved_memory_ops')) return 'other';
    if (name.startsWith('vmi_type_attr_parse')) return 'other';
    if (name.startsWith('vmi_op_verifier_basic')) return 'other';
    return 'verifier';
}

function notesFor(name, cat) {
    if (cat === 'lowering') {
        if (name.includes('_invalid')) return 'Lowering VMI to VPTO - invalid input rejection';
        return 'Lowering VMI to VPTO ops';
    }
    if (cat === 'layout') {
        if (name.includes('_invalid')) return 'Layout assignment - invalid input rejection';
        return 'Layout assignment pass test';
    }
    if (cat === 'gate') {
        if (name.includes('_invalid')) return 'Layout gate - invalid input rejection';
        return 'Layout gate validation test';
    }
    if (cat === 'verifier') {
        if (name.includes('_invalid')) return 'Verifier rejection test (invalid input)';
        if (name.includes('_valid')) return 'Verifier acceptance test (valid input)';
        return 'Verifier test';
    }
    if (cat === 'other') {
        if (name.includes('ptoas')) return 'PTOAS CLI/integration test';
        if (name.includes('legalize')) return 'Legalization pass test';
        if (name.includes('producer_boundary')) return 'Producer boundary check test';
        if (name.includes('interleaved')) return 'Interleaved memory operations test';
        return 'Other VMI test';
    }
    return '';
}

const results = [];
for (const fname of filenames) {
    const cat = categorize(fname);
    const ops = extractOps(fname);
    const notes = notesFor(fname, cat);
    results.push({
        filename: fname + '.pto',
        category: cat,
        ops: ops,
        notes: notes
    });
}

// Count by category
const byCat = {};
for (const r of results) {
    byCat[r.category] = (byCat[r.category] || 0) + 1;
}

const output = {
    files: results,
    total: results.length,
    byCategory: byCat
};

const outPath = 'C:/Users/53284/Desktop/ptocode/PTOAS/test/lit/vmi_new/vmi_catalog.json';
fs.writeFileSync(outPath, JSON.stringify(output, null, 2));
console.log(`Total: ${results.length}`);
console.log(`By category: ${JSON.stringify(byCat)}`);
console.log(`Written to: ${outPath}`);

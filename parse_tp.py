from hta.trace_analysis import TraceAnalysis

trace_dir = "./hta_traces"
analyzer = TraceAnalysis(trace_dir=trace_dir)
time_spent_df = analyzer.get_temporal_breakdown(visualize=True)
print(time_spent_df)
#time_spent_df.plot()

idle_time_df = analyzer.get_idle_time_breakdown(ranks={0,1}, show_idle_interval_stats=False)
print(idle_time_df)

overlap_df = analyzer.get_comm_comp_overlap()
print(overlap_df)

#analyzer.generate_trace_with_counters()

kernel_info_df = analyzer.get_cuda_kernel_launch_stats()
print(kernel_info_df)

kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown()
print(kernel_metrics_df)


workload(sycl::queue q1) 
{
    sycl::queue q2(inroder());

    sycl::event event1 = workload_a(q1);
    sycl::event event2 = workload_b(q2, depend: event1);
    sycl::event event3 = workload_c(q1, depend: event1);
    workload_d(q1, depend: {event2, event3});
}

main()
{
    syclgraph g;
    sycl::queue q1(inroder());
    sycl::queue q3(inroder());
    g.begin_recording();
    workload(q1);
    g.end_recording();
    execg = g.finalize();
    execg.replay(q1 or q3);
}

{
    
if (q1.is_recording()) {
    g = q1.get_graph()
    // relax the sycl graph restriction
    sycl::event event2 = g.add_external_engine_work(
            [&](...){workload_b_new(...);}, q2, depend: event1);
}
else
    sycl::event event2 = workload_b(q2, depend: event1);


SyclGraph::add_external_engine_work(T external_binding_function, sycl::queue q, vector<sycl::event> dependent_events)
{
    // do not execute,
    // but to record it as a node of the sycl graph
    
    sycl::event return_event;  // not tied with backend event yet
    sycl::event event2 = q.get_last_event();  // in-order queue
    
    sycl::graph::node node(external_binding_function, 
                           dependencies: {dependent_events, event2}, 
                           signal: return_event, 
                           type=external_engine_work);
    this->addnode(node);    
    q.set_external_event(return_event)
    return return_event;
}

SyclGraph::finalize()
{
    for (node in this->nodes) {
        if (node.type == external_engine_work) {  // workload b
            sycl::event events = node.get_dependencies()
            sycl::event return_event = node.get_signal();
            levelzero::event lz_event = zeCreateEvent()
            return_event.from_native_event(lz_event);  // now tie the sycl event with backend event
            node.offloading = node.external_binding_function(depend: {events.get_ptr()}, signal: return_event.get_ptr());
        } else {
            // workload a, c and d are recorded into one sycl graph as usual
            ...
        }
    }
}

SyclGraph::replay()
{
    // make sure the events between the sycl graph and external_engine_work are reset
    
    // workload a, c and d are submitted to FS1 computation as usual
    submit_command_for_sycl_graph();  // zeCommandQueueExecuteCommandLists ...
    
    // workload b is submitted to the external engine (FS1 communication)
    for node in this->nodes if node.type == external_engine_work
        node.offloading();

    // let's see what happened after all the commands are submitted
    1. computation hardware is executing workload a, communication hardware is waiting
    2. computation hardware finishes workload a, and then:
        computation hardware starts workload c
        communication hardware starts workload b
    3. after both workloads b and c are finished, computation hardware starts workload d    
}
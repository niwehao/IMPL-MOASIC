/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "PerfModel/EventList.h"
#include <cassert>

using namespace Simulation;

EventList::EventList(const EventTime event_time) noexcept : event_time(event_time) {
    assert(event_time >= 0);
    events = std::priority_queue<Event, std::vector<Event>, EventPrioCompare>();
}

EventTime EventList::get_event_time() const noexcept {
    return event_time;
}

void EventList::add_event(const Callback callback, const CallbackArg callback_arg, unsigned int priority) noexcept {
    assert(callback != nullptr);
    events.emplace(callback, callback_arg, priority);
}

void EventList::invoke_events() noexcept {
    // invoke all events in the event list
    while (!events.empty()) {
        auto e = events.top();
        events.pop();
        e.invoke_event();
    }
}

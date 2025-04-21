
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get(self, sid):
        return self.sessions.setdefault(sid, {"intent": None, "step": None, "data": {}})

    def clear(self, sid):
        self.sessions.pop(sid, None)

session_manager = SessionManager()


from abc import ABC, abstractmethod

class BaseTask(ABC):
    @abstractmethod
    async def handle(self, sid, sio, session, payload):
        pass


from tasks.base import BaseTask

class CreateTicket(BaseTask):
    async def handle(self, sid, sio, session, payload):
        step = session['step']
        
        if step is None:
            await sio.emit('bot_uttered', {
                'message': 'I can help you create a support ticket. Please enter your GCI ID:',
                'template': 'enter_gci_template'
            }, to=sid)
            session['step'] = 'awaiting_gci'

        elif step == 'awaiting_gci':
            gci_id = payload.get('gci_id')
            if gci_id:
                session['data']['gci_id'] = gci_id
                cases = ['CASE123', 'CASE456']  # Replace with actual DB query
                await sio.emit('bot_uttered', {
                    'message': 'Please select your case:',
                    'cases': cases,
                    'template': 'select_case_template'
                }, to=sid)
                session['step'] = 'awaiting_case_selection'

        elif step == 'awaiting_case_selection':
            selected_case = payload.get('selected_case_id')
            if selected_case:
                session['data']['selected_case'] = selected_case
                await sio.emit('bot_uttered', {
                    'message': f"You've selected case {selected_case}. Confirm ticket creation?",
                    'options': ['Yes', 'No']
                }, to=sid)
                session['step'] = 'confirm_ticket'

        elif step == 'confirm_ticket':
            confirmation = payload.get('confirmation')
            if confirmation == 'Yes':
                case = session['data']['selected_case']
                await sio.emit('bot_uttered', {
                    'message': f'Ticket for case {case} created successfully!'
                }, to=sid)
                session.clear()
            else:
                await sio.emit('bot_uttered', {
                    'message': 'Ticket creation canceled. How else can I assist?'
                }, to=sid)
                session.clear()



from tasks.create_ticket import CreateTicket
# Import other tasks here...

TASKS = {
    "create_ticket": CreateTicket(),
    # "get_case_details": GetCaseDetails(),
    # Add other tasks similarly...
}

def get_task(intent):
    return TASKS.get(intent)







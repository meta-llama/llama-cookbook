list_emails_function = """
{
    "type": "function",
    "function": {
        "name": "list_emails",
        "description": "Return a list of emails matching an optionally specified query.",
        "parameters": {
            "type": "dic",
            "properties": [
                {
                    "max_results": {
                        "type": "integer",
                        "description": "The default maximum number of emails to return is 100; the maximum allowed value for this field is 500."
                    }
                },              
                {
                    "query": {
                        "type": "string",
                        "description": "One or more keywords in the email subject and body, or one or more filters. There can be 6 types of filters: 1) Field-specific Filters: from, to, cc, bcc, subject; 2) Date Filters: before, after, older than, newer than); 3) Status Filters: read, unread, starred, importatant; 4) Attachment Filters: has, filename or type; 5) Size Filters: larger, smaller; 6) logical operators (or, and, not)."
                    }
                }
            ],
            "required": []
        }
    }
}
"""

get_email_function = """
{
    "type": "function",
    "function": {
        "name": "get_email_detail",
        "description": "Get detailed info about a specific email",
        "parameters": {
            "type": "dict",
            "properties": [
                {
                    "detail": {
                        "type": "string",
                        "description": "what detail the user wants to know about - two possible values: body or attachment"
                    }
                },
                {
                    "which": {
                        "type": "string",
                        "description": "which email to get detail about - possible values include: 'first', 'second', ..., 'last', 'from ...', and 'subject ...'"
                    }
                },
            ],
            "required": ["detail", "which"]
        }
    }
}
"""

send_email_function = """
{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Compose, reply, or forward email",
        "parameters": {
            "type": "dict",
            "properties": [
                {
                    "action": {
                        "type": "string",
                        "description": "Whether to compose, reply, or forward an email"
                    }
                },
                {
                    "to": {
                        "type": "string",
                        "description": "The recipient of the email"
                    }
                },
                {
                    "subject": {
                        "type": "string",
                        "description": "The email subject"
                    }
                },
                {
                    "body": {
                        "type": "string",
                        "description": "The email content"
                    }
                },                                
                {
                    "email_id": {
                        "type": "string",
                        "description": "the email id to reply or forward to"
                    }
                }
            ],
            "required": ["action", "to", "subject", "body"]
        }
    }
}
"""

get_pdf_summary_function = """
{
    "type": "function",
    "function": {
        "name": "get_pdf_summary",
        "description": "get a summary of a PDF attachment",
        "parameters": {
            "type": "dict",
            "properties": [
                {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the PDF file"
                    }
                },
            ],
            "required": ["file_name"]
        }
    }
}
"""

create_draft_function = """
{
    "type": "function",
    "function": {
        "name": "create_draft",
        "description": "Create a new, reply, or forward email draft",
        "parameters": {
            "type": "dict",
            "properties": [
                {
                    "action": {
                        "type": "string",
                        "description": "Whether to draft a new, reply, or forward an email"
                    }
                },
                {
                    "to": {
                        "type": "string",
                        "description": "The recipient of the email"
                    }
                },
                {
                    "subject": {
                        "type": "string",
                        "description": "The email subject"
                    }
                },
                {
                    "body": {
                        "type": "string",
                        "description": "The email content"
                    }
                },                                
                {
                    "email_id": {
                        "type": "string",
                        "description": "the email id to reply or forward to, or empty if draft a new email."
                    }
                }
            ],
            "required": ["action", "to", "subject", "body", "email_id"]
        }
    }
}
"""

# for now, only allow for one draft email to be saved in a session
# to support for multiple drafts, cf how get_email_detail after list_emails is implemented.
send_draft_function = """
{
    "type": "function",
    "function": {
        "name": "send_draft",
        "description": "Send a draft email",
        "parameters": {
            "type": "dict",
            "properties": [
                {
                    "id": {
                        "type": "string",
                        "description": "draft id"
                    }
                },        
            ],
            "required": ["id"]        
        }
    }
}
"""

examples = """
{"name": "list_emails", "parameters": {"query": "has:attachment larger:5mb"}}
{"name": "list_emails", "parameters": {"query": "has:attachment"}}
{"name": "list_emails", "parameters": {"query": "newer_than:1d"}}
{"name": "list_emails", "parameters": {"query": "older_than:1d"}}
{"name": "list_emails", "parameters": {"query": "is:unread"}}
{"name": "list_emails", "parameters":  {"query": "<query> is:unread"}}
{"name": "list_emails", "parameters":  {"query": "<query> is:read"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "first"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "last"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "second"}}
{"name": "get_email_detail", "parameters": {"detail": "body", "which": "subject <subject info>"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "from <sender info>"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "first"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "last"}}
{"name": "get_email_detail", "parameters": {"detail": "attachment", "which": "<email id>"}}
{"name": "send_email", "parameters": {"action": "compose", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx"}}
{"name": "send_email", "parameters": {"action": "reply", "to": "", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "send_email", "parameters": {"action": "forward", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "create_draft", "parameters": {"action": "new", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx", "email_id": ""}}
{"name": "create_draft", "parameters": {"action": "reply", "to": "", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "create_draft", "parameters": {"action": "forward", "to": "jeffxtang@meta.com", "subject": "xxxxx", "body": "xxxxx", "email_id": "xxxxx"}}
{"name": "send_draft", "parameters": {"id": "..."}}
{"name": "get_pdf_summary", "parameters": {"file_name": "..."}}
"""

system_prompt = f"""
Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 1 December 2024

Your name is Email Agent, an assistant that can perform all email related tasks for your user.
Respond to the user's ask by making use of the following functions if needed.
If no available functions can be used, just say "I don't know" and don't make up facts.
Here is a list of available functions in JSON format:

{list_emails_function}
{get_email_function}
{send_email_function}
{get_pdf_summary_function}
{create_draft_function}
{send_draft_function}

Example responses:
{examples}

"""
